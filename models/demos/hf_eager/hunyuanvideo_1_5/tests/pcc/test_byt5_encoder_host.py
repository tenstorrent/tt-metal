# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only correctness evidence for the on-device HunyuanVideo-1.5 byT5 encoder.

Everything here runs on CPU and never opens a Tenstorrent device. It covers the
parts of the port whose correctness does not depend on kernel numerics:

* placement gating (which meshes are legal for a 6-head / 1472-wide encoder),
* the checkpoint contract and the state-dict name/shape mapping onto the tt_dit
  module tree, including the tensor-parallel shard shapes,
* the relative-position bucketing the port reimplements,
* the host-side tokenization/padding/masking the adapter performs, and
* an op-for-op torch mirror of the TT dataflow (independent q/k/v width, TP
  fracture and gather) checked against HuggingFace's own byT5 forward.

Kernel-level numerics -- bf16 matmuls, the TTNN softmax, and the `(mask-1)*inf`
additive-mask expression the shared T5 stack uses -- are only observable on
hardware; see `test_byt5_encoder_pcc.py`.
"""

from __future__ import annotations

import glob
import json
import math
import os
import types

import pytest
import torch

from models.demos.hf_eager.hunyuanvideo_1_5.tt.byt5_encoder import (
    DEFAULT_PROMPT_LENGTH,
    analyze_byt5_support,
    byt5_cache_name,
    byt5_tt_config,
    finalize_byt5_output,
    plan_byt5_inputs,
    require_byt5_support,
    select_byt5_device,
)

_SNAPSHOT_GLOB = "models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_*"

# The real geometry, restated here so the tests fail loudly if the adapter's
# expectations are edited without revisiting this file.
D_MODEL = 1472
D_FF = 3584
D_KV = 64
NUM_HEADS = 6
NUM_LAYERS = 12
VOCAB = 1510
INNER = NUM_HEADS * D_KV  # 384, deliberately != D_MODEL


def _hunyuan_byt5_config(**overrides):
    values = dict(
        architectures=["T5EncoderModel"],
        d_model=D_MODEL,
        d_ff=D_FF,
        d_kv=D_KV,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        vocab_size=VOCAB,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        layer_norm_epsilon=1e-6,
        feed_forward_proj="gated-gelu",
        dense_act_fn="gelu_new",
        is_encoder_decoder=False,
        is_gated_act=True,
        tie_word_embeddings=False,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _snapshot_subdir(name):
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    matches = sorted(glob.glob(os.path.join(hub, _SNAPSHOT_GLOB, "snapshots", "*", name)))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Placement gating
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mesh_shape", "strategy"),
    [((1, 1), "TP1-axis1"), ((1, 2), "TP2-axis1"), ((2, 1), "TP2-axis0")],
)
def test_byt5_accepts_only_dedicated_one_and_two_device_meshes(mesh_shape, strategy):
    support = analyze_byt5_support(_hunyuan_byt5_config(), mesh_shape)
    assert support.supported
    assert support.strategy == strategy
    assert support.tensor_parallel == (1 if strategy.startswith("TP1") else 2)
    assert support.mesh_axis == int(strategy[-1])
    assert f"independent width {INNER}" in support.reason
    assert f"d_model={D_MODEL}" in support.reason


@pytest.mark.parametrize("mesh_shape", [(8, 4), (1, 4), (2, 2), (4, 8), (1, 8)])
def test_byt5_fails_closed_on_meshes_tensor_parallelism_cannot_divide(mesh_shape):
    support = analyze_byt5_support(_hunyuan_byt5_config(), mesh_shape)
    assert not support.supported
    assert support.strategy == "host"
    # d_model = 1472 = 2^6 * 23 and num_heads = 6, so no factor above 2 divides both.
    assert "no factor above 2 is legal" in support.reason
    with pytest.raises(RuntimeError, match="Host byT5 remains the correct default"):
        require_byt5_support(_hunyuan_byt5_config(), mesh_shape)


@pytest.mark.parametrize(
    ("override", "needle"),
    [
        ({"d_kv": 128}, "d_kv=128"),
        ({"d_model": 1536}, "d_model=1536"),
        ({"vocab_size": 384}, "vocab_size=384"),
        ({"num_layers": 24}, "num_layers=24"),
        ({"dense_act_fn": "relu"}, "dense_act_fn='relu'"),
        ({"is_gated_act": False}, "is_gated_act=False"),
        ({"architectures": ["T5Model"]}, "architectures="),
    ],
)
def test_byt5_fails_closed_on_checkpoint_variants(override, needle):
    support = analyze_byt5_support(_hunyuan_byt5_config(**override), (1, 2))
    assert not support.supported
    assert needle in support.reason


def test_tie_word_embeddings_is_reported_but_never_rejected():
    """The one field the strict contract deliberately does not enforce.

    `text_encoder_2/config.json` stores `false`, but `T5Config.from_pretrained`
    returns `True` -- HuggingFace does not round-trip it. Enforcing it rejected
    the real checkpoint before any hardware work, and the field cannot matter:
    it ties an LM head to the input embedding, and `T5EncoderModel` has no LM
    head. With the checkpoint's own value restored, all five hardware PCC cases
    pass (TP1 0.999935, TP2 0.999938, full sequence 0.999931). A checkpoint that
    genuinely carried an LM head is still rejected, by the strict unexpected-key
    check in `load_torch_state_dict`.

    The disagreement is surfaced in `reason` rather than silently dropped.
    """
    tied = analyze_byt5_support(_hunyuan_byt5_config(tie_word_embeddings=True), (1, 2))
    assert tied.supported, tied.reason
    assert tied.strategy == "TP2-axis1"
    assert "tie_word_embeddings=True" in tied.reason
    assert "does not round-trip" in tied.reason

    untied = analyze_byt5_support(_hunyuan_byt5_config(tie_word_embeddings=False), (1, 2))
    assert untied.supported
    assert "tie_word_embeddings" not in untied.reason

    # Any *other* field is still fail-closed, so this is one exemption and not
    # a general loosening of the contract.
    assert not analyze_byt5_support(_hunyuan_byt5_config(tie_word_embeddings=True, d_ff=4096), (1, 2)).supported


def test_select_byt5_device_never_adopts_the_dit_mesh():
    config = _hunyuan_byt5_config()
    device, support = select_byt5_device(config, types.SimpleNamespace(shape=(8, 4)))
    assert device is None
    assert not support.supported

    legal = types.SimpleNamespace(shape=(1, 2))
    device, support = select_byt5_device(config, legal)
    assert device is legal and support.supported

    device, support = select_byt5_device(config, None)
    assert device is None
    assert "HY_BYT5_SUBMESH" in support.reason


# ---------------------------------------------------------------------------
# Checkpoint contract
# ---------------------------------------------------------------------------


def test_real_checkpoint_config_matches_the_supported_contract():
    directory = _snapshot_subdir("text_encoder_2")
    if directory is None:
        pytest.skip("no local HunyuanVideo-1.5 snapshot with a text_encoder_2 directory")
    with open(os.path.join(directory, "config.json")) as handle:
        config = types.SimpleNamespace(**json.load(handle))

    support = analyze_byt5_support(config, (1, 2))
    assert support.supported, support.reason

    tt_config = byt5_tt_config(config)
    assert tt_config.embed_dim == D_MODEL
    assert tt_config.ff_dim == D_FF
    assert tt_config.num_heads == NUM_HEADS
    assert tt_config.kv_dim == D_KV
    assert tt_config.attention_inner_dim == INNER != tt_config.embed_dim
    assert tt_config.max_prompt_length == DEFAULT_PROMPT_LENGTH
    # Only the first T5 layer carries a learned relative-position bias; the rest
    # reuse it. UMT5 (the other user of this shared stack) differs here.
    assert tt_config.use_relative_position_bias == [True] + [False] * (NUM_LAYERS - 1)
    assert byt5_cache_name(config) == f"HunyuanVideo-1.5-byT5-v{VOCAB}-h{D_MODEL}-f{D_FF}-a{INNER}-l{NUM_LAYERS}"


def test_the_parsed_real_config_is_accepted_even_though_hf_rewrites_a_field():
    """Reproduce the round-trip quirk on the actual checkpoint, without silicon.

    `test_real_checkpoint_config_matches_the_supported_contract` reads the raw
    JSON; the adapter in production is handed whatever `from_pretrained`
    returns. Those two disagree on `tie_word_embeddings`, which is how the
    hardware PCC gate came to fail closed on a valid checkpoint, so check the
    parsed object explicitly.
    """
    directory = _snapshot_subdir("text_encoder_2")
    if directory is None:
        pytest.skip("no local HunyuanVideo-1.5 snapshot with a text_encoder_2 directory")
    from transformers import T5Config

    with open(os.path.join(directory, "config.json")) as handle:
        stored = json.load(handle)
    parsed = T5Config.from_pretrained(directory, local_files_only=True)

    support = analyze_byt5_support(parsed, (1, 2))
    assert support.supported, support.reason
    print(
        f"byT5 config.json tie_word_embeddings={stored.get('tie_word_embeddings')!r}; "
        f"T5Config.from_pretrained -> {parsed.tie_word_embeddings!r}",
        flush=True,
    )


def _expected_hf_state_shapes():
    """The `T5EncoderModel.state_dict()` key/shape set implied by the config.

    `shared.weight` and `encoder.embed_tokens.weight` are the same tensor; both
    keys are emitted because `T5EncoderModel` holds the embedding twice.
    """
    shapes = {
        "shared.weight": (VOCAB, D_MODEL),
        "encoder.embed_tokens.weight": (VOCAB, D_MODEL),
        "encoder.final_layer_norm.weight": (D_MODEL,),
    }
    for layer in range(NUM_LAYERS):
        prefix = f"encoder.block.{layer}."
        shapes[f"{prefix}layer.0.SelfAttention.q.weight"] = (INNER, D_MODEL)
        shapes[f"{prefix}layer.0.SelfAttention.k.weight"] = (INNER, D_MODEL)
        shapes[f"{prefix}layer.0.SelfAttention.v.weight"] = (INNER, D_MODEL)
        shapes[f"{prefix}layer.0.SelfAttention.o.weight"] = (D_MODEL, INNER)
        shapes[f"{prefix}layer.0.layer_norm.weight"] = (D_MODEL,)
        shapes[f"{prefix}layer.1.DenseReluDense.wi_0.weight"] = (D_FF, D_MODEL)
        shapes[f"{prefix}layer.1.DenseReluDense.wi_1.weight"] = (D_FF, D_MODEL)
        shapes[f"{prefix}layer.1.DenseReluDense.wo.weight"] = (D_MODEL, D_FF)
        shapes[f"{prefix}layer.1.layer_norm.weight"] = (D_MODEL,)
    shapes["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = (32, NUM_HEADS)
    return shapes


def test_expected_state_shapes_match_the_real_safetensors_checkpoint():
    directory = _snapshot_subdir("text_encoder_2")
    if directory is None:
        pytest.skip("no local HunyuanVideo-1.5 snapshot with a text_encoder_2 directory")
    path = os.path.join(directory, "model.safetensors")
    if not os.path.isfile(path):
        pytest.skip("text_encoder_2/model.safetensors is not present in the local snapshot")

    from safetensors import safe_open

    with safe_open(path, "pt") as handle:
        actual = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}

    expected = _expected_hf_state_shapes()
    # The serialized file stores the shared embedding once; `state_dict()` adds
    # the tied `encoder.embed_tokens.weight` alias back.
    expected.pop("encoder.embed_tokens.weight")
    assert actual == expected


class _FakeMeshDevice:
    """Enough of `ttnn.MeshDevice` to build the module tree without silicon.

    Only `shape` (parameter sharding and the tensor-parallel guards) and `arch`
    (`ttnn.init_device_compute_kernel_config`, a pure host call) are touched
    during construction and weight loading.
    """

    def __init__(self, shape):
        self.shape = tuple(shape)

    def arch(self):
        import ttnn

        return ttnn.device.Arch.BLACKHOLE


def _named_parameters(module, prefix=""):
    for name, child in module.named_children():
        yield from _named_parameters(child, f"{prefix}{name}.")
    for name, parameter in module.named_parameters():
        yield f"{prefix}{name}", parameter


@pytest.mark.parametrize(("mesh_shape", "tensor_parallel"), [((1, 1), 1), ((1, 2), 2)])
def test_state_dict_names_and_tensor_parallel_shards_map_onto_the_tt_module_tree(
    monkeypatch, mesh_shape, tensor_parallel
):
    """Load the real checkpoint's key/shape set into the real module tree.

    Weight *uploads* are stubbed out (they need a device); everything else --
    `_prepare_torch_state` renaming, strict missing/unexpected-key checking, the
    `Parameter` shard-divisibility arithmetic and the per-parameter global-shape
    check -- is the production code path.
    """
    import ttnn
    from models.tt_dit.encoders.t5.model_t5 import T5Encoder
    from models.tt_dit.layers.module import LoadingError, Parameter
    from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor

    loaded = {}

    def _record(self, torch_tensor, /):
        shape = tuple(torch_tensor.shape)
        if shape != self.total_shape:
            raise LoadingError(f"expected tensor shape {self.total_shape}, got {shape}")
        loaded[id(self)] = shape

    monkeypatch.setattr(Parameter, "load_torch_tensor", _record)

    device = _FakeMeshDevice(mesh_shape)
    axis = 1 if mesh_shape[1] == tensor_parallel else 0
    parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=tensor_parallel, mesh_axis=axis))
    model = T5Encoder(byt5_tt_config(_hunyuan_byt5_config()), device, None, parallel_config)

    state = {
        key: torch.empty(shape, device="meta", dtype=torch.float32)
        for key, shape in _expected_hf_state_shapes().items()
    }
    incompatible = model.load_torch_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    parameters = dict(_named_parameters(model))
    assert all(id(parameter) in loaded for parameter in parameters.values()), "some parameters were never loaded"

    # The transposed, tensor-parallel-fractured local shapes each device holds.
    expected_local = {
        "token_embeddings.weight": (VOCAB, D_MODEL),  # replicated: byte-level table, never sharded
        "final_layer_norm.weight": (1, D_MODEL),
        "encoder.layers.0.self_attn.relative_attention_bias.weight": (32, NUM_HEADS // tensor_parallel),
    }
    for layer in range(NUM_LAYERS):
        prefix = f"encoder.layers.{layer}."
        for projection in ("q_proj", "k_proj", "v_proj"):
            expected_local[f"{prefix}self_attn.{projection}.weight"] = (D_MODEL, INNER // tensor_parallel)
        expected_local[f"{prefix}self_attn.o_proj.weight"] = (INNER, D_MODEL // tensor_parallel)
        expected_local[f"{prefix}self_attn.layer_norm.weight"] = (1, D_MODEL)
        expected_local[f"{prefix}ff.layer_norm.weight"] = (1, D_MODEL)
        expected_local[f"{prefix}ff.dense_gated_dense.wi0.weight"] = (D_MODEL, D_FF // tensor_parallel)
        expected_local[f"{prefix}ff.dense_gated_dense.wi1.weight"] = (D_MODEL, D_FF // tensor_parallel)
        expected_local[f"{prefix}ff.dense_gated_dense.wo.weight"] = (D_FF // tensor_parallel, D_MODEL)

    assert set(parameters) == set(expected_local)
    for name, expected in expected_local.items():
        assert parameters[name].local_shape == expected, name

    # The byte-level embedding table must stay replicated: it is the one weight
    # whose row count (1510) is not tile aligned, and `ttnn.embedding` reads it
    # in ROW_MAJOR layout.
    embedding = parameters["token_embeddings.weight"]
    assert embedding.mesh_axes == (None, None)
    assert embedding.layout == ttnn.ROW_MAJOR_LAYOUT


def test_state_dict_load_is_strict_about_unexpected_keys(monkeypatch):
    from models.tt_dit.encoders.t5.model_t5 import T5Encoder
    from models.tt_dit.layers.module import Parameter
    from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor

    monkeypatch.setattr(Parameter, "load_torch_tensor", lambda self, t: None)
    model = T5Encoder(
        byt5_tt_config(_hunyuan_byt5_config()),
        _FakeMeshDevice((1, 2)),
        None,
        EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=2, mesh_axis=1)),
    )
    state = {key: torch.empty(shape, device="meta") for key, shape in _expected_hf_state_shapes().items()}
    state["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.bias"] = torch.empty(
        (NUM_HEADS,), device="meta"
    )
    with pytest.raises(ValueError, match="unexpected Torch state keys"):
        model.load_torch_state_dict(state, strict=True)


# ---------------------------------------------------------------------------
# Relative position bias
# ---------------------------------------------------------------------------


def test_relative_position_bucketing_matches_huggingface():
    """The port reimplements T5's bucketing; it must agree exactly with HF."""
    from transformers.models.t5.modeling_t5 import T5Attention

    from models.tt_dit.encoders.t5.model_t5 import _relative_position_bucket

    length = DEFAULT_PROMPT_LENGTH
    context = torch.arange(length)[:, None]
    memory = torch.arange(length)[None, :]
    relative_position = memory - context

    actual = _relative_position_bucket(relative_position, num_buckets=32, max_distance=128)
    reference = T5Attention._relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    )
    torch.testing.assert_close(actual, reference)
    assert int(actual.max()) < 32 and int(actual.min()) >= 0


# ---------------------------------------------------------------------------
# Host-side preprocessing
# ---------------------------------------------------------------------------


def test_plan_byt5_inputs_is_a_no_op_at_the_pipeline_window():
    ids = torch.randint(0, VOCAB, (1, DEFAULT_PROMPT_LENGTH))
    mask = torch.cat([torch.ones(1, 22), torch.zeros(1, DEFAULT_PROMPT_LENGTH - 22)], dim=1)
    planned_ids, planned_mask, length = plan_byt5_inputs(ids, mask, vocab_size=VOCAB)
    assert length == DEFAULT_PROMPT_LENGTH
    assert planned_ids.dtype == torch.int32
    assert planned_mask.dtype == torch.bfloat16
    torch.testing.assert_close(planned_ids, ids.to(torch.int32))
    torch.testing.assert_close(planned_mask.float(), mask)


def test_plan_byt5_inputs_pads_to_a_tile_and_masks_the_synthesized_tail():
    ids = torch.randint(1, VOCAB, (2, 40))
    planned_ids, planned_mask, length = plan_byt5_inputs(ids, None, vocab_size=VOCAB)
    assert length == 40
    assert planned_ids.shape == (2, 64)
    # A mask is synthesized even though the caller passed none, so the real
    # tokens cannot attend to the padding this function introduced.
    assert planned_mask is not None
    assert planned_mask[:, :40].float().eq(1).all()
    assert planned_mask[:, 40:].float().eq(0).all()
    assert planned_ids[:, 40:].eq(0).all()


def test_plan_byt5_inputs_rejects_inputs_that_would_silently_corrupt_the_lookup():
    with pytest.raises(ValueError, match="out of range"):
        plan_byt5_inputs(torch.tensor([[VOCAB]]), None, vocab_size=VOCAB)
    with pytest.raises(ValueError, match="out of range"):
        plan_byt5_inputs(torch.tensor([[-1]]), None, vocab_size=VOCAB)
    with pytest.raises(ValueError, match="rank 2"):
        plan_byt5_inputs(torch.zeros(4, dtype=torch.long), None, vocab_size=VOCAB)
    with pytest.raises(ValueError, match="does not match"):
        plan_byt5_inputs(torch.zeros(1, 8, dtype=torch.long), torch.ones(1, 4), vocab_size=VOCAB)
    with pytest.raises(ValueError, match="only 0 and 1"):
        plan_byt5_inputs(torch.zeros(1, 4, dtype=torch.long), torch.full((1, 4), 0.5), vocab_size=VOCAB)


def test_finalize_crops_the_tile_padding_and_neutralizes_masked_positions():
    embedding = torch.randn(1, 64, 8)
    mask = torch.cat([torch.ones(1, 10), torch.zeros(1, 54)], dim=1)

    cropped = finalize_byt5_output(embedding, length=40, attention_mask=mask, zero_padding=False)
    assert cropped.shape == (1, 40, 8)
    torch.testing.assert_close(cropped, embedding[:, :40])

    zeroed = finalize_byt5_output(embedding, length=40, attention_mask=mask, zero_padding=True)
    torch.testing.assert_close(zeroed[:, :10], embedding[:, :10])
    assert zeroed[:, 10:].eq(0).all()


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def test_glyph_extraction_and_tokenization_stay_inside_the_checkpoint_vocabulary():
    directory = _snapshot_subdir("tokenizer_2")
    if directory is None:
        pytest.skip("no local HunyuanVideo-1.5 snapshot with a tokenizer_2 directory")
    from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5 import extract_glyph_texts
    from transformers import AutoTokenizer

    # No quoted span means the pipeline never calls byT5 at all: it emits zeros
    # and an all-zero mask, so the device path is only exercised by glyph prompts.
    assert extract_glyph_texts("a cat walks on the grass") is None
    glyph = extract_glyph_texts('a neon sign reading "OPEN" beside another reading "24/7"')
    assert glyph == 'Text "OPEN". Text "24/7". '

    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    tokens = tokenizer(
        glyph,
        padding="max_length",
        max_length=DEFAULT_PROMPT_LENGTH,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    ids, mask = tokens.input_ids, tokens.attention_mask
    assert ids.shape == (1, DEFAULT_PROMPT_LENGTH)
    assert int(ids.max()) < VOCAB
    # The mask is a contiguous valid prefix, which is what the DiT's
    # `_trim_to_valid` padding isolation relies on.
    valid = int(mask.sum())
    assert mask[0, :valid].eq(1).all() and mask[0, valid:].eq(0).all()
    # And the adapter accepts it unchanged.
    planned_ids, planned_mask, length = plan_byt5_inputs(ids, mask.float(), vocab_size=VOCAB)
    assert length == DEFAULT_PROMPT_LENGTH and planned_ids.shape == ids.shape
    assert int(planned_mask.float().sum()) == valid


# ---------------------------------------------------------------------------
# Torch mirror of the TT dataflow
# ---------------------------------------------------------------------------


def _t5_rms_norm(x, weight, eps):
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


def _tt_dataflow_reference(hf_model, input_ids, attention_mask, *, tensor_parallel):
    """Mirror `models/tt_dit/encoders/t5/model_t5.py` op-for-op in torch.

    Reproduces, in the same order the TT graph executes them: the replicated
    embedding lookup, the layer-0-only relative position bias with the additive
    attention mask folded in, per-device column-parallel q/k/v at the independent
    inner width, the local head split, the gather back to the full inner width
    before the output projection, the column-parallel output projection gathered
    back to `d_model`, and the row-parallel FFN whose partial products are summed
    across devices.
    """
    from models.tt_dit.encoders.t5.model_t5 import _relative_position_bucket

    config = hf_model.config
    state = hf_model.state_dict()
    eps = config.layer_norm_epsilon
    inner = config.num_heads * config.d_kv
    local_heads = config.num_heads // tensor_parallel
    local_inner = inner // tensor_parallel
    local_model = config.d_model // tensor_parallel
    local_ff = config.d_ff // tensor_parallel

    def shard_out(weight, index, width):
        """Column-parallel: `Linear._prepare_torch_state` transposes (out, in) to
        (in, out) and the mesh fractures the output columns."""
        return weight.t()[:, index * width : (index + 1) * width]

    def shard_in(weight, index, width):
        """Row-parallel: transposed to (in, out), fractured over input rows."""
        return weight.t()[index * width : (index + 1) * width, :]

    hidden = torch.nn.functional.embedding(input_ids, state["encoder.embed_tokens.weight"])
    length = input_ids.shape[-1]

    context = torch.arange(length)[:, None]
    buckets = _relative_position_bucket(
        torch.arange(length)[None, :] - context,
        num_buckets=config.relative_attention_num_buckets,
        max_distance=config.relative_attention_max_distance,
    )
    bias_table = state["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]
    position_bias = torch.nn.functional.embedding(buckets, bias_table).permute(2, 0, 1).unsqueeze(0)
    if attention_mask is not None:
        # `T5Stack.forward` turns the 0/1 mask into an additive bias and folds it
        # into the layer-0 position bias, which every later layer reuses. A large
        # finite negative stands in for the device expression's infinity so this
        # host mirror cannot produce 0*inf NaNs; both drive the softmax to zero.
        additive = (attention_mask.to(position_bias.dtype) - 1.0) * 1.0e9
        position_bias = position_bias + additive.reshape(attention_mask.shape[0], 1, 1, -1)

    for layer in range(config.num_layers):
        prefix = f"encoder.block.{layer}."
        residual = hidden
        normed = _t5_rms_norm(hidden, state[f"{prefix}layer.0.layer_norm.weight"], eps)

        gathered_attention = []
        for device in range(tensor_parallel):
            q = normed @ shard_out(state[f"{prefix}layer.0.SelfAttention.q.weight"], device, local_inner)
            k = normed @ shard_out(state[f"{prefix}layer.0.SelfAttention.k.weight"], device, local_inner)
            v = normed @ shard_out(state[f"{prefix}layer.0.SelfAttention.v.weight"], device, local_inner)
            shape = (*q.shape[:-1], local_heads, config.d_kv)
            q = q.view(shape).transpose(1, 2)
            k = k.view(shape).transpose(1, 2)
            v = v.view(shape).transpose(1, 2)
            # T5 folds the 1/sqrt(d) scaling into initialization; there is none here.
            scores = q @ k.transpose(-1, -2)
            scores = scores + position_bias[:, device * local_heads : (device + 1) * local_heads]
            out = torch.softmax(scores, dim=-1) @ v
            gathered_attention.append(out.transpose(1, 2).reshape(*normed.shape[:-1], local_inner))
        attention = torch.cat(gathered_attention, dim=-1)

        projected = torch.cat(
            [
                attention @ shard_out(state[f"{prefix}layer.0.SelfAttention.o.weight"], device, local_model)
                for device in range(tensor_parallel)
            ],
            dim=-1,
        )
        hidden = residual + projected

        residual = hidden
        normed = _t5_rms_norm(hidden, state[f"{prefix}layer.1.layer_norm.weight"], eps)
        partials = []
        for device in range(tensor_parallel):
            gate = torch.nn.functional.gelu(
                normed @ shard_out(state[f"{prefix}layer.1.DenseReluDense.wi_0.weight"], device, local_ff),
                approximate="tanh",
            )
            up = normed @ shard_out(state[f"{prefix}layer.1.DenseReluDense.wi_1.weight"], device, local_ff)
            partials.append(
                (gate * up) @ shard_in(state[f"{prefix}layer.1.DenseReluDense.wo.weight"], device, local_ff)
            )
        hidden = residual + sum(partials)

    return _t5_rms_norm(hidden, state["encoder.final_layer_norm.weight"], eps)


@pytest.mark.parametrize("tensor_parallel", [1, 2])
def test_tt_dataflow_matches_huggingface_byt5_at_the_independent_attention_width(tensor_parallel):
    """Algebraic proof that the port's decomposition is exact, without silicon.

    Uses HunyuanVideo's real byT5 widths (`d_model=1472`, `d_ff=3584`, 6 heads,
    `d_kv=64`, so the attention inner width is 384) with a reduced layer count and
    vocabulary so the test stays cheap. What matters is the width relationship,
    which is what the shared T5 port had to be extended to support.
    """
    from transformers import T5Config as HFT5Config
    from transformers import T5EncoderModel

    torch.manual_seed(0)
    config = HFT5Config(
        vocab_size=512,
        d_model=D_MODEL,
        d_ff=D_FF,
        d_kv=D_KV,
        num_heads=NUM_HEADS,
        num_layers=2,
        num_decoder_layers=0,
        feed_forward_proj="gated-gelu",
        dense_act_fn="gelu_new",
        is_encoder_decoder=False,
        tie_word_embeddings=False,
        layer_norm_epsilon=1e-6,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.0,
    )
    model = T5EncoderModel(config).eval()

    tokens = torch.randint(0, config.vocab_size, (2, 64))
    mask = torch.zeros(2, 64)
    mask[0, :37] = 1
    mask[1, :5] = 1

    with torch.no_grad():
        reference = model(input_ids=tokens, attention_mask=mask).last_hidden_state
        actual = _tt_dataflow_reference(model, tokens, mask, tensor_parallel=tensor_parallel)

    # Only the masked-in positions are meaningful; HF leaves the rest to whatever
    # the residual stream produced and the DiT drops them via the mask.
    selector = mask.bool().unsqueeze(-1).expand_as(reference)
    torch.testing.assert_close(actual[selector], reference[selector], rtol=2e-4, atol=2e-4)


def test_gated_gelu_uses_the_tanh_approximation_the_checkpoint_declares():
    """`dense_act_fn='gelu_new'` is the tanh approximation, and the port's
    `activation_fn='gelu_tanh'` fuses the matching TTNN unary. Exact erf GELU
    would be a silent numerical mismatch, so pin the distinction here."""
    from transformers.activations import ACT2FN

    x = torch.linspace(-4, 4, 257)
    torch.testing.assert_close(ACT2FN["gelu_new"](x), torch.nn.functional.gelu(x, approximate="tanh"))
    assert not torch.allclose(ACT2FN["gelu_new"](x), torch.nn.functional.gelu(x), atol=1e-5)


def test_additive_mask_and_hard_masking_agree_after_softmax():
    """`T5Stack` folds the 0/1 mask into the position bias as a large negative
    additive term (HuggingFace uses `finfo.min`, the TT stack uses an infinity).
    Both must reduce to renormalizing over the valid keys only."""
    torch.manual_seed(0)
    scores = torch.randn(2, 3, 8, 8)
    mask = torch.zeros(2, 8)
    mask[0, :5] = 1
    mask[1, :1] = 1
    additive = ((1.0 - mask) * torch.finfo(torch.float32).min).reshape(2, 1, 1, 8)

    approximate = torch.softmax(scores + additive, dim=-1)
    exact = torch.softmax(scores.masked_fill(~mask.bool().reshape(2, 1, 1, 8), -math.inf), dim=-1)
    torch.testing.assert_close(approximate, exact)
    assert approximate[..., 5:].sum() == 0
