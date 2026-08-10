# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Llama safetensors loader."""

from __future__ import annotations

from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

import ttnn
import ttml
from ttml.models import EmbeddingPlacement, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig, load_from_safetensors
from ttml.models.llama.safetensors_loader import (
    ROW_DIM,
    _TIED_NAMES,
    _assemble,
    _canonical,
    _check_coverage,
    _pad_to,
    _require_shape,
    _rules,
    _sharded_dim,
    _to_bf16_4d,
    _unpermute_proj_rows,
)
from ttml.testing import read_mesh_tensor

TP_AXIS_SIZE = 2  # the 'tp' extent of conftest's tp_mesh fixture
HEAD_DIM = 8  # even: RoPE splits each head into two halves
HIDDEN = 16

# (num_heads, num_kv_heads, tp_size) -- GQA, then MHA (one group per head) and MQA (one group).
QKV_CASES = [
    (8, 4, 1),
    (8, 4, 2),
    (8, 4, 4),
    (16, 8, 8),
    (8, 8, 2),
    (8, 1, 1),
]


def fuse_qkv(q, k, v, num_heads, num_kv_heads, tp_size):
    """What ``_rules`` asks the driver to do for ``qkv_linear``, without the device."""
    blocks = [_unpermute_proj_rows(q, num_heads), _unpermute_proj_rows(k, num_kv_heads), v]
    return _assemble(blocks, ROW_DIM, tp_size, "qkv_linear")


def fuse_gate_up(gate, up, tp_size):
    return _assemble([gate, up], ROW_DIM, tp_size, "w_gate_up")


def hf_qkv(num_heads: int, num_kv_heads: int, seed: int):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((num_heads * HEAD_DIM, HIDDEN)).astype(np.float32),
        rng.standard_normal((num_kv_heads * HEAD_DIM, HIDDEN)).astype(np.float32),
        rng.standard_normal((num_kv_heads * HEAD_DIM, HIDDEN)).astype(np.float32),
    )


def unfuse(fused: np.ndarray, block_rows: list[int], tp_size: int) -> list[np.ndarray]:
    """Inverse of the interleave."""
    per_rank = [rows // tp_size for rows in block_rows]
    shard_rows = sum(per_rank)
    assert fused.shape[0] == shard_rows * tp_size, "fused row count does not match the blocks"

    recovered: list[list[np.ndarray]] = [[] for _ in block_rows]
    for rank in range(tp_size):
        shard = fused[rank * shard_rows : (rank + 1) * shard_rows]
        offset = 0
        for i, rows in enumerate(per_rank):
            recovered[i].append(shard[offset : offset + rows])
            offset += rows
    return [np.concatenate(parts, axis=0) for parts in recovered]


class TestFuseQkv:
    @pytest.mark.parametrize("num_heads,num_kv_heads,tp_size", QKV_CASES)
    def test_width_matches_model(self, num_heads, num_kv_heads, tp_size):
        q, k, v = hf_qkv(num_heads, num_kv_heads, seed=0)
        fused = fuse_qkv(q, k, v, num_heads, num_kv_heads, tp_size)
        assert fused.shape == ((num_heads + 2 * num_kv_heads) * HEAD_DIM, HIDDEN)

    @pytest.mark.parametrize("num_heads,num_kv_heads,tp_size", QKV_CASES)
    def test_every_shard_holds_matched_q_k_v(self, num_heads, num_kv_heads, tp_size):
        q, k, v = hf_qkv(num_heads, num_kv_heads, seed=1)
        fused = fuse_qkv(q, k, v, num_heads, num_kv_heads, tp_size)

        block_rows = [num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM, num_kv_heads * HEAD_DIM]
        got_q, got_k, got_v = unfuse(fused, block_rows, tp_size)

        assert np.array_equal(got_q, _unpermute_proj_rows(q, num_heads)), "Q rows scrambled"
        assert np.array_equal(got_k, _unpermute_proj_rows(k, num_kv_heads)), "K rows scrambled"
        assert np.array_equal(got_v, v), "V rows scrambled"

    @pytest.mark.parametrize("num_heads,num_kv_heads,tp_size", QKV_CASES)
    def test_each_rank_holds_the_groups_its_heads_need(self, num_heads, num_kv_heads, tp_size):
        """GQA runs locally over ``num_groups // tp`` groups, so co-locating a head with its
        group is what makes local attention equal the global one."""
        q, k, v = hf_qkv(num_heads, num_kv_heads, seed=2)
        fused = fuse_qkv(q, k, v, num_heads, num_kv_heads, tp_size)

        heads_per_group = num_heads // num_kv_heads
        local_heads, local_groups = num_heads // tp_size, num_kv_heads // tp_size
        shard_rows = (local_heads + 2 * local_groups) * HEAD_DIM
        unpermuted_k = _unpermute_proj_rows(k, num_kv_heads)
        assert fused.shape[0] == shard_rows * tp_size, "fused height disagrees with the local block sizes"

        for rank in range(tp_size):
            shard = fused[rank * shard_rows : (rank + 1) * shard_rows]
            local_k = shard[local_heads * HEAD_DIM :][: local_groups * HEAD_DIM]
            for local_head in range(local_heads):
                global_head = rank * local_heads + local_head
                local_group = local_head // heads_per_group
                global_group = global_head // heads_per_group
                assert np.array_equal(
                    local_k[local_group * HEAD_DIM : (local_group + 1) * HEAD_DIM],
                    unpermuted_k[global_group * HEAD_DIM : (global_group + 1) * HEAD_DIM],
                ), f"rank {rank} head {local_head} reads the wrong K group"

    @pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 4), (8, 8), (8, 1)])
    def test_tp1_is_a_plain_block_concat(self, num_heads, num_kv_heads):
        q, k, v = hf_qkv(num_heads, num_kv_heads, seed=3)
        expected = np.concatenate(
            [_unpermute_proj_rows(q, num_heads), _unpermute_proj_rows(k, num_kv_heads), v], axis=0
        )
        assert np.array_equal(fuse_qkv(q, k, v, num_heads, num_kv_heads, 1), expected)

    def test_global_block_concat_is_wrong_above_tp1(self):
        """Guard against regressing to the plain concat: at tp>1 it hands rank 0 only Q and K."""
        num_heads, num_kv_heads, tp_size = 8, 4, 2
        q, k, v = hf_qkv(num_heads, num_kv_heads, seed=4)
        naive = np.concatenate([_unpermute_proj_rows(q, num_heads), _unpermute_proj_rows(k, num_kv_heads), v], axis=0)
        assert not np.array_equal(fuse_qkv(q, k, v, num_heads, num_kv_heads, tp_size), naive)


class TestFuseGateUp:
    @pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
    def test_every_shard_holds_matched_gate_up(self, tp_size):
        intermediate = 256
        rng = np.random.default_rng(5)
        gate = rng.standard_normal((intermediate, HIDDEN)).astype(np.float32)
        up = rng.standard_normal((intermediate, HIDDEN)).astype(np.float32)

        fused = fuse_gate_up(gate, up, tp_size)
        assert fused.shape == (2 * intermediate, HIDDEN)

        got_gate, got_up = unfuse(fused, [intermediate, intermediate], tp_size)
        assert np.array_equal(got_gate, gate), "gate rows scrambled"
        assert np.array_equal(got_up, up), "up rows scrambled"

    def test_tp1_is_a_plain_concat(self):
        rng = np.random.default_rng(6)
        gate, up = rng.standard_normal((64, HIDDEN)), rng.standard_normal((64, HIDDEN))
        assert np.array_equal(fuse_gate_up(gate, up, 1), np.concatenate([gate, up], axis=0))


class TestAssemble:
    """Interleaving follows from the destination's placement, not from the caller."""

    @pytest.mark.parametrize("shard_dim", [None, 3], ids=["replicated", "cols-sharded"])
    def test_only_row_sharding_interleaves(self, shard_dim):
        rng = np.random.default_rng(7)
        a, b = rng.standard_normal((8, 4)), rng.standard_normal((8, 4))
        plain = np.concatenate([a, b], axis=0)
        assert np.array_equal(_assemble([a, b], shard_dim, 4, "w"), plain)

    def test_single_block_is_untouched(self):
        block = np.random.default_rng(8).standard_normal((8, 4))
        assert _assemble([block], ROW_DIM, 4, "w") is block

    def test_rejects_rows_not_divisible_over_the_mesh(self, expect_error):
        block = np.zeros((6, HIDDEN), np.float32)
        with expect_error(RuntimeError, "not divisible over 4 devices"):
            _assemble([block, block], ROW_DIM, 4, "w_gate_up")

    def test_replicated_blocks_need_not_divide_the_mesh(self):
        """Only a row-shard splits, so an indivisible replicated weight is fine."""
        block = np.zeros((3, HIDDEN), np.float32)
        assert _assemble([block, block], None, 4, "w_gate_up").shape == (6, HIDDEN)

    @pytest.mark.parametrize("shard_dim", [ROW_DIM, 3, None], ids=["rows", "cols", "replicated"])
    def test_rejects_mismatched_widths_whatever_the_placement(self, shard_dim, expect_error):
        """Stacking on rows needs one width; nothing upstream of _assemble checks per-source shapes."""
        blocks = [np.zeros((4, HIDDEN), np.float32), np.zeros((4, HIDDEN + 1), np.float32)]
        with expect_error(RuntimeError, "columns, expected"):
            _assemble(blocks, shard_dim, 2, "qkv_linear")


class TestPlacementWithoutTensorParallelism:
    """Deriving placement puts the no-mesh and no-tp-axis cases on the hot path for plain
    single-device loads (GRPO, inference), even though nothing is sharded there.
    """

    def test_no_mesh_open_is_replicated(self, monkeypatch):
        monkeypatch.setattr(ttml, "maybe_mesh", lambda: None)
        assert _sharded_dim(object(), "w") is None

    def test_mesh_without_a_tp_axis_is_replicated(self, monkeypatch):
        monkeypatch.setattr(ttml, "maybe_mesh", lambda: SimpleNamespace(has_axis=lambda _: False))
        assert _sharded_dim(object(), "w") is None


class TestHelpers:
    @pytest.mark.parametrize(
        "raw,canonical",
        [
            ("model.embed_tokens.weight", "embed_tokens.weight"),
            ("embed_tokens.weight", "embed_tokens.weight"),
            ("model.wte.weight", "embed_tokens.weight"),
            ("transformer.wte.weight", "embed_tokens.weight"),
            ("model.layers.3.self_attn.q_proj.weight", "layers.3.self_attn.q_proj.weight"),
            ("layers.3.self_attn.q_proj.weight", "layers.3.self_attn.q_proj.weight"),
            ("lm_head.weight", "lm_head.weight"),
        ],
    )
    def test_canonical_collapses_checkpoint_spellings(self, raw, canonical):
        assert _canonical(raw) == canonical

    @pytest.mark.parametrize("shape", [(4, 3), (5, 4)])
    def test_require_shape_rejects_a_mismatch(self, shape, expect_error):
        with expect_error(RuntimeError, "but the parameter is"):
            _require_shape(np.zeros(shape, np.float32), (3, 4), "w")

    def test_pad_keeps_the_checkpoint_block_and_is_deterministic(self):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        first, second = _pad_to(arr, (4, 3), "w"), _pad_to(arr, (4, 3), "w")
        assert np.array_equal(first[:2], arr), "checkpoint rows moved"
        assert np.array_equal(first, second), "padding differs between loads of one checkpoint"
        assert first[2:].any(), "padding is all zeros, which leaves dead neurons"

    def test_pad_refuses_to_discard_weights(self, expect_error):
        """Silently cropping an oversized checkpoint would throw away vocabulary."""
        with expect_error(RuntimeError, "would discard weights"):
            _pad_to(np.zeros((8, 3), np.float32), (4, 3), "w")

    def test_unpermute_round_trips_at_four_rows_per_head(self):
        """Self-inverse only up to 4 rows per head, and at 2 it is the identity -- so 4 is the one
        size where the round trip says anything about the permutation.
        """
        w = np.arange(16, dtype=np.float32).reshape(8, 2)
        once = _unpermute_proj_rows(w, 2)
        assert not np.array_equal(once, w), "unpermute moved nothing"
        assert np.array_equal(_unpermute_proj_rows(once, 2), w)

    def test_to_bf16_4d_forces_c_order(self):
        """from_numpy reads the buffer linearly and ignores strides, so an F-ordered view would
        reach the device transposed."""
        transposed = np.arange(12, dtype=np.float32).reshape(3, 4).T
        out = _to_bf16_4d(transposed)
        assert out.flags.c_contiguous, "astype(order='K') kept the source layout"
        assert np.array_equal(out.reshape(4, 3).astype(np.float32), transposed)

    def test_unpermute_matches_the_explicit_definition(self):
        """out[2i] = w[i], out[2i+1] = w[half+i], per head."""
        n_heads, per_head, cols = 3, 6, 2
        w = np.arange(n_heads * per_head * cols, dtype=np.float32).reshape(n_heads * per_head, cols)
        got, half = _unpermute_proj_rows(w, n_heads), per_head // 2
        for head in range(n_heads):
            base = head * per_head
            for i in range(half):
                assert np.array_equal(got[base + 2 * i], w[base + i])
                assert np.array_equal(got[base + 2 * i + 1], w[base + half + i])


# ── Coverage: does the rule set match the model? ──

_TOP_LEVEL = ("Llama/tok_emb/weight", "Llama/fc/weight", "Llama/ln_fc/gamma")
_PER_LAYER_FUSED = (
    "attention_norm/gamma",
    "mlp_norm/gamma",
    "attention/qkv_linear/weight",
    "attention/out_linear/weight",
    "mlp/w_gate_up/weight",
    "mlp/w2/weight",
)
_PER_LAYER_PRE_FUSION = (
    "attention_norm/gamma",
    "mlp_norm/gamma",
    "attention/q_linear/weight",
    "attention/kv_linear/weight",
    "attention/out_linear/weight",
    "mlp/w1/weight",
    "mlp/w3/weight",
    "mlp/w2/weight",
)


N_LAYERS = 2
N_HEADS = 4
N_KV_HEADS = 2
MODEL_HIDDEN = 128
INTERMEDIATE = 128
VOCAB = 128
SEQ_LEN = 64
MODEL_HEAD_DIM = MODEL_HIDDEN // N_HEADS


def param_names(per_layer, num_layers=N_LAYERS):
    return {*_TOP_LEVEL, *(f"Llama/blocks/{i}/{p}" for i in range(num_layers) for p in per_layer)}


def coverage_config(**overrides):
    return LlamaConfig(
        hidden_size=MODEL_HIDDEN,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        num_hidden_layers=N_LAYERS,
        intermediate_size=INTERMEDIATE,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ_LEN,
        **overrides,
    )


class TestCoverage:
    """Device-free: ``_rules`` is pure and ``_check_coverage`` takes a set of names."""

    def test_accepts_the_current_model(self):
        config, names = coverage_config(), param_names(_PER_LAYER_FUSED)
        _check_coverage(names, list(_rules(config, names)))

    def test_rejects_the_pre_fusion_model(self, expect_error):
        """The break this loader was fixed for: the model fused q/k/v and gate/up while the
        rules still named q_linear/kv_linear/w1/w3."""
        config, names = coverage_config(), param_names(_PER_LAYER_PRE_FUSION)
        with expect_error(RuntimeError, "disagree about its parameters"):
            _check_coverage(names, list(_rules(config, names)))

    def test_names_both_directions_of_a_disagreement(self, expect_error):
        config, names = coverage_config(), param_names(_PER_LAYER_PRE_FUSION)
        with expect_error(RuntimeError, "disagree about its parameters") as excinfo:
            _check_coverage(names, list(_rules(config, names)))
        message = str(excinfo.value)
        assert "no rule feeds       Llama/blocks/0/attention/q_linear/weight" in message
        assert "no such parameter   Llama/blocks/0/attention/qkv_linear/weight" in message

    def test_attention_biases_are_declared_init_only(self):
        """HF Llama ships no attention biases, so a biased model must still be loadable."""
        config = coverage_config(attention_bias=True)
        names = param_names(_PER_LAYER_FUSED) | {
            f"Llama/blocks/{i}/attention/{linear}/bias"
            for i in range(N_LAYERS)
            for linear in ("qkv_linear", "out_linear")
        }
        _check_coverage(names, list(_rules(config, names)))

    @pytest.mark.parametrize("survivor", ["Llama/fc/weight", "Llama/tok_emb/weight"])
    def test_tying_feeds_whichever_name_survived(self, survivor):
        """Which of the two names survives dedup is a module-traversal detail, not a contract."""
        config = coverage_config(weight_tying=WeightTyingType.Enabled)
        names = (param_names(_PER_LAYER_FUSED) - set(_TIED_NAMES)) | {survivor}
        targets = [rule.param for rule in _rules(config, names)]
        assert targets.count(survivor) == 1, "tied embedding must be written once"
        _check_coverage(names, list(_rules(config, names)))

    def test_tying_rejects_two_separate_parameters(self, expect_error):
        """Both names present means unshared tensors, so one would keep its init values."""
        config, names = coverage_config(weight_tying=WeightTyingType.Enabled), param_names(_PER_LAYER_FUSED)
        with expect_error(RuntimeError, "exactly one of"):
            list(_rules(config, names))


# ── End-to-end: a synthetic HF checkpoint through the real loader ──


def e2e_config(use_tp: bool, placement: EmbeddingPlacement) -> LlamaConfig:
    return coverage_config(
        use_tp=use_tp,
        embedding_placement=placement,
        weight_tying=WeightTyingType.Disabled,
    )


def write_hf_checkpoint(directory) -> dict[str, np.ndarray]:
    save_file = pytest.importorskip("safetensors.numpy").save_file

    rng = np.random.default_rng(99)

    def w(rows, cols):
        return rng.standard_normal((rows, cols)).astype(np.float32)

    tensors = {
        "model.embed_tokens.weight": w(VOCAB, MODEL_HIDDEN),
        "lm_head.weight": w(VOCAB, MODEL_HIDDEN),
        "model.norm.weight": rng.standard_normal(MODEL_HIDDEN).astype(np.float32),
    }
    for layer in range(N_LAYERS):
        pfx = f"model.layers.{layer}"
        tensors.update(
            {
                f"{pfx}.input_layernorm.weight": rng.standard_normal(MODEL_HIDDEN).astype(np.float32),
                f"{pfx}.post_attention_layernorm.weight": rng.standard_normal(MODEL_HIDDEN).astype(np.float32),
                f"{pfx}.self_attn.q_proj.weight": w(N_HEADS * MODEL_HEAD_DIM, MODEL_HIDDEN),
                f"{pfx}.self_attn.k_proj.weight": w(N_KV_HEADS * MODEL_HEAD_DIM, MODEL_HIDDEN),
                f"{pfx}.self_attn.v_proj.weight": w(N_KV_HEADS * MODEL_HEAD_DIM, MODEL_HIDDEN),
                f"{pfx}.self_attn.o_proj.weight": w(MODEL_HIDDEN, MODEL_HIDDEN),
                f"{pfx}.mlp.gate_proj.weight": w(INTERMEDIATE, MODEL_HIDDEN),
                f"{pfx}.mlp.up_proj.weight": w(INTERMEDIATE, MODEL_HIDDEN),
                f"{pfx}.mlp.down_proj.weight": w(MODEL_HIDDEN, INTERMEDIATE),
            }
        )
    save_file(tensors, str(directory / "model.safetensors"))
    return tensors


def as_bf16(array: np.ndarray) -> np.ndarray:
    """*array* as the loader stores it, so a read-back comparison can be exact."""
    return array.astype(ml_dtypes.bfloat16).astype(np.float32)


def read_param(params, name: str, concat_dims: dict[str, int] | None = None) -> np.ndarray:
    """A parameter as a 2-D ``[out_features, in_features]`` array, gathered over the mesh."""
    replica_dim = 1
    # Replicated: name the axis rather than lean on the composer's default axis-to-dim mapping,
    # which only puts 'tp' on dim 1 while 'tp' happens to be the second mesh axis.
    gathered = read_mesh_tensor(params[name], concat_dims or {"tp": replica_dim})
    if concat_dims is None:
        # The copies are identical; assert the count first, or the comparison below is a no-op.
        assert gathered.shape[replica_dim] == TP_AXIS_SIZE, f"{name}: expected {TP_AXIS_SIZE} replicas"
        for copy in range(1, gathered.shape[replica_dim]):
            assert np.array_equal(gathered[:, copy], gathered[:, 0]), f"{name}: replicas disagree"
        gathered = gathered[:, :1]
    return gathered.reshape(gathered.shape[-2], gathered.shape[-1])


def read_replicated(tensor, stack_dim: int) -> np.ndarray:
    """One copy of a replicated tensor; the composer materializes it once per device."""
    parts = np.split(read_mesh_tensor(tensor, {"tp": stack_dim}), TP_AXIS_SIZE, axis=stack_dim)
    for i, part in enumerate(parts[1:], 1):
        assert np.array_equal(part, parts[0]), f"replica {i} differs from replica 0"
    return parts[0]


@pytest.mark.requires_device
@pytest.mark.usefixtures("tp_mesh")
class TestConsumerContract:
    """A wrong block order here still yields finite, varying logits -- silently wrong
    attention. (``swiglu_packed``'s half-split: test_swiglu_packed.py.)
    """

    def test_heads_creation_reads_q_then_k_then_v_contiguously_by_head(self):
        batch, seq, heads, groups, head_dim = 1, 32, 4, 2, 32

        # A distinct value per head pins the block order *and* the head order within a block.
        def block(count, base):
            values = base + np.arange(count, dtype=np.float32)
            return np.repeat(values, head_dim).reshape(1, 1, 1, count * head_dim) * np.ones(
                (batch, 1, seq, 1), np.float32
            )

        q, k, v = block(heads, 10.0), block(groups, 100.0), block(groups, 200.0)
        qkv = ttml.autograd.Tensor.from_numpy(
            np.concatenate([q, k, v], axis=3), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
        )

        got_q, got_k, got_v = ttml.ops.multi_head_utils.heads_creation(qkv, heads, groups)

        blocks = (("Q", got_q, heads, 10.0), ("K", got_k, groups, 100.0), ("V", got_v, groups, 200.0))
        for label, got, count, base in blocks:
            values = read_replicated(got, 3)
            assert values.shape == (batch, count, seq, head_dim), f"{label}: shape {values.shape}"
            for head in range(count):
                expected = base + head
                assert (values[:, head] == expected).all(), (
                    f"{label} head {head} holds {np.unique(values[:, head])}, expected {expected}: "
                    f"the fused width is not read as [Q|K|V] contiguous by head"
                )


@pytest.mark.requires_device
@pytest.mark.usefixtures("tp_mesh")
class TestCoverageAgainstRealModels:
    @pytest.mark.parametrize(
        "use_tp,placement,tying",
        [
            (False, EmbeddingPlacement.Replicated, WeightTyingType.Disabled),
            (False, EmbeddingPlacement.Replicated, WeightTyingType.Enabled),
            (True, EmbeddingPlacement.Replicated, WeightTyingType.Disabled),
            (True, EmbeddingPlacement.VocabParallel, WeightTyingType.Disabled),
            (True, EmbeddingPlacement.VocabParallel, WeightTyingType.Enabled),
            (True, EmbeddingPlacement.FeatureParallel, WeightTyingType.Disabled),
        ],
        ids=lambda v: getattr(v, "name", str(v)),
    )
    def test_rules_cover_the_model(self, use_tp, placement, tying):
        config = coverage_config(use_tp=use_tp, embedding_placement=placement, weight_tying=tying)
        names = set(Llama(config).parameters())
        _check_coverage(names, list(_rules(config, names)))

    def test_biased_attention_is_coverable(self):
        config = coverage_config(use_tp=True, attention_bias=True)
        names = set(Llama(config).parameters())
        _check_coverage(names, list(_rules(config, names)))


@pytest.mark.requires_device
@pytest.mark.usefixtures("tp_mesh")
class TestLoadIntoModel:
    @pytest.mark.parametrize(
        "placement",
        [EmbeddingPlacement.Replicated, EmbeddingPlacement.VocabParallel, EmbeddingPlacement.FeatureParallel],
        ids=lambda p: p.name,
    )
    def test_tensor_parallel_layout(self, tmp_path, placement):
        hf = write_hf_checkpoint(tmp_path)
        config = e2e_config(True, placement)
        model = Llama(config)
        load_from_safetensors(model, tmp_path, config)

        params = model.parameters()
        for layer in range(N_LAYERS):
            pfx = f"model.layers.{layer}"
            expected_qkv = fuse_qkv(
                hf[f"{pfx}.self_attn.q_proj.weight"],
                hf[f"{pfx}.self_attn.k_proj.weight"],
                hf[f"{pfx}.self_attn.v_proj.weight"],
                N_HEADS,
                N_KV_HEADS,
                TP_AXIS_SIZE,
            )
            got = read_param(params, f"Llama/blocks/{layer}/attention/qkv_linear/weight", {"tp": 2})
            assert np.array_equal(got, as_bf16(expected_qkv)), f"layer {layer}: qkv_linear layout"

            expected_gate_up = fuse_gate_up(
                hf[f"{pfx}.mlp.gate_proj.weight"], hf[f"{pfx}.mlp.up_proj.weight"], TP_AXIS_SIZE
            )
            got = read_param(params, f"Llama/blocks/{layer}/mlp/w_gate_up/weight", {"tp": 2})
            assert np.array_equal(got, as_bf16(expected_gate_up)), f"layer {layer}: w_gate_up layout"

            # Row-parallel weights shard the input features (dim 3), unfused and uninterleaved.
            got = read_param(params, f"Llama/blocks/{layer}/attention/out_linear/weight", {"tp": 3})
            assert np.array_equal(got, as_bf16(hf[f"{pfx}.self_attn.o_proj.weight"])), f"layer {layer}: o_proj"
            got = read_param(params, f"Llama/blocks/{layer}/mlp/w2/weight", {"tp": 3})
            assert np.array_equal(got, as_bf16(hf[f"{pfx}.mlp.down_proj.weight"])), f"layer {layer}: down_proj"

        embedding_dims = {
            EmbeddingPlacement.Replicated: None,
            EmbeddingPlacement.VocabParallel: {"tp": 2},
            EmbeddingPlacement.FeatureParallel: {"tp": 3},
        }[placement]
        got = read_param(params, "Llama/tok_emb/weight", embedding_dims)
        assert np.array_equal(got, as_bf16(hf["model.embed_tokens.weight"])), "token embedding"

    def test_reports_a_clean_load(self, tmp_path, capsys):
        hf = write_hf_checkpoint(tmp_path)
        config = e2e_config(True, EmbeddingPlacement.VocabParallel)
        model = Llama(config)
        load_from_safetensors(model, tmp_path, config)

        report = capsys.readouterr().out
        # Positive first: the absences below also hold for a report that was never printed.
        assert f"Loaded {len(model.parameters())} parameters from {len(hf)} checkpoint tensors" in report, report
        assert "were not used" not in report, report
        assert "Left at initial values" not in report, report

    def test_rejects_a_checkpoint_missing_a_fused_source(self, tmp_path, expect_error):
        """A fused parameter needs all of its sources; a partial group must not load silently."""
        save_file = pytest.importorskip("safetensors.numpy").save_file
        tensors = {k: v for k, v in write_hf_checkpoint(tmp_path).items() if "v_proj" not in k}
        (tmp_path / "model.safetensors").unlink()
        save_file(tensors, str(tmp_path / "model.safetensors"))

        config = e2e_config(True, EmbeddingPlacement.VocabParallel)
        with expect_error(RuntimeError, "the checkpoint has no"):
            load_from_safetensors(Llama(config), tmp_path, config)

    def test_without_tensor_parallelism_is_a_plain_concat(self, tmp_path):
        hf = write_hf_checkpoint(tmp_path)
        config = e2e_config(False, EmbeddingPlacement.Replicated)
        model = Llama(config)
        load_from_safetensors(model, tmp_path, config)

        params = model.parameters()
        pfx = "model.layers.0"
        expected_qkv = np.concatenate(
            [
                _unpermute_proj_rows(hf[f"{pfx}.self_attn.q_proj.weight"], N_HEADS),
                _unpermute_proj_rows(hf[f"{pfx}.self_attn.k_proj.weight"], N_KV_HEADS),
                hf[f"{pfx}.self_attn.v_proj.weight"],
            ],
            axis=0,
        )
        got = read_param(params, "Llama/blocks/0/attention/qkv_linear/weight")
        assert np.array_equal(got, as_bf16(expected_qkv)), "qkv_linear without TP"

        expected_gate_up = np.concatenate([hf[f"{pfx}.mlp.gate_proj.weight"], hf[f"{pfx}.mlp.up_proj.weight"]], axis=0)
        got = read_param(params, "Llama/blocks/0/mlp/w_gate_up/weight")
        assert np.array_equal(got, as_bf16(expected_gate_up)), "w_gate_up without TP"

    def test_forward_runs_on_loaded_weights(self, tmp_path):
        """Loaded weights must actually drive the fused ops, not just sit at the right shape."""
        write_hf_checkpoint(tmp_path)
        config = e2e_config(True, EmbeddingPlacement.VocabParallel)
        model = Llama(config)
        load_from_safetensors(model, tmp_path, config)
        model.eval()

        ids = np.random.default_rng(7).integers(0, VOCAB, (1, 1, 1, SEQ_LEN)).astype(np.uint32)
        mask = np.tril(np.ones((1, 1, SEQ_LEN, SEQ_LEN), dtype=np.float32))
        logits = model(
            ttml.autograd.Tensor.from_numpy(ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32),
            ttml.autograd.Tensor.from_numpy(mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16),
        )

        values = read_mesh_tensor(logits, {"tp": 3})
        assert np.isfinite(values).all(), "logits contain non-finite values"
        assert values.std() > 1e-3, "logits are ~constant; the weights did not reach the matmuls"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
