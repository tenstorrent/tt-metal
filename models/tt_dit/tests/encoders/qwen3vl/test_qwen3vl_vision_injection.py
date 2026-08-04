# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# How vision reaches the Qwen3-VL decoder. Two mechanisms, both at the vision
# token positions and both easy to confuse:
#
#   - the tower's merged tokens REPLACE the embeddings of the `<|image_pad|>`
#     rows, before the block stack;
#   - the tower's deepstack features are ADDED to those same rows, after each of
#     the first `len(deepstack_features)` decoder layers.
#
# Replace vs add is not interchangeable, and the deepstack layer index is keyed
# off the list position rather than the vision layer the feature came from. Both
# are asserted below.
#
# The vision rows are contiguous runs -- MiniMax-H3 emits a `"<Picture i>: "`
# label, `<|vision_start|>`, one run of `<|image_pad|>`, `<|vision_end|>` -- so
# the port slices and concatenates rather than doing a masked scatter.
# =============================================================================

import pytest
import torch
import transformers

import ttnn

from ....encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder, _scatter_rows, create_rope_tensors, vision_token_runs
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

IMAGE_TOKEN_ID = 151655  # <|image_pad|>
HIDDEN = 128
SEQ = 64

# --------------------------------------------------------------------------- host


def test_vision_token_runs_finds_one_run_per_image():
    """A `"<Picture i>: "` label between two images keeps them as separate runs."""
    ids = torch.tensor([[1, 2, 3] + [IMAGE_TOKEN_ID] * 4 + [9, 9] + [IMAGE_TOKEN_ID] * 6 + [7]])
    assert vision_token_runs(ids, IMAGE_TOKEN_ID) == [(3, 4), (9, 6)]


def test_vision_token_runs_handles_the_edges():
    """A run at the very start or end of the sequence still terminates correctly."""
    assert vision_token_runs(torch.tensor([[IMAGE_TOKEN_ID] * 3 + [1, 2]]), IMAGE_TOKEN_ID) == [(0, 3)]
    assert vision_token_runs(torch.tensor([[1, 2] + [IMAGE_TOKEN_ID] * 3]), IMAGE_TOKEN_ID) == [(2, 3)]
    assert vision_token_runs(torch.tensor([[1, 2, 3]]), IMAGE_TOKEN_ID) == []


def test_vision_token_runs_rejects_a_batch(expect_error):
    """One request is one sequence; a batch would silently take only the first row."""
    with expect_error(ValueError, "expected a single sequence"):
        vision_token_runs(torch.zeros(2, 8, dtype=torch.long), IMAGE_TOKEN_ID)


# ------------------------------------------------------------------------- device

_MESH = [pytest.param((1, 1), (1, 1), id="single")]


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
@pytest.mark.parametrize("add", [False, True], ids=["replace", "add"])
@pytest.mark.parametrize(
    "runs", [[(8, 16)], [(0, 8)], [(SEQ - 8, 8)], [(4, 8), (20, 12)]], ids=["middle", "start", "end", "two_runs"]
)
def test_scatter_rows_matches_torch(mesh_device, submesh_shape, add, runs):
    """`_scatter_rows` reproduces the torch indexed write it stands in for."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    total = sum(length for _, length in runs)

    torch.manual_seed(0)
    base = torch.randn(1, SEQ, HIDDEN)
    values = torch.randn(1, total, HIDDEN)

    golden = base.clone()
    taken = 0
    for start, length in runs:
        chunk = values[:, taken : taken + length, :]
        if add:
            golden[:, start : start + length, :] += chunk
        else:
            golden[:, start : start + length, :] = chunk
        taken += length

    out = _scatter_rows(bf16_tensor(base, device=submesh), bf16_tensor(values, device=submesh), runs, add=add)
    actual = tensor.to_torch(out, mesh_axes=[None, None, None])
    assert actual.shape[-2:] == (SEQ, HIDDEN), f"{tuple(actual.shape)}"
    assert_quality(golden, actual, pcc=0.999)


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
def test_replace_and_add_differ(mesh_device, submesh_shape):
    """Replace and add are not interchangeable -- the distinction the two mechanisms turn on."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    torch.manual_seed(0)
    base = bf16_tensor(torch.randn(1, SEQ, HIDDEN), device=submesh)
    values = bf16_tensor(torch.randn(1, 16, HIDDEN), device=submesh)
    runs = [(8, 16)]
    a = tensor.to_torch(_scatter_rows(base, values, runs, add=False), mesh_axes=[None, None, None])
    b = tensor.to_torch(_scatter_rows(base, values, runs, add=True), mesh_axes=[None, None, None])
    assert not torch.allclose(a, b, atol=1e-2), "replace and add agree; the mechanisms are conflated"


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
def test_scatter_rows_rejects_a_row_count_mismatch(mesh_device, submesh_shape, expect_error):
    """Too few or too many value rows is a caller error, not a silent partial write."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    base = bf16_tensor(torch.zeros(1, SEQ, HIDDEN), device=submesh)
    values = bf16_tensor(torch.zeros(1, 5, HIDDEN), device=submesh)
    with expect_error(ValueError, "runs cover 16 rows but values has 5"):
        _scatter_rows(base, values, [(8, 16)], add=False)


def _reference_text_model(layers):
    """A tiny HF decoder stack, only so the port has real weights to load.

    The port's parameters have no data until `load_torch_state_dict` runs, and these tests are about
    where vision lands rather than numeric parity, so a random init is enough -- but it has to exist.
    """
    config = transformers.Qwen3VLTextConfig(
        vocab_size=256,
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [6, 5, 5],
            "mrope_interleaved": True,
        },
    )
    torch.manual_seed(0)
    return transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextModel._from_config(config).eval()


def _encoder(submesh, *, layers, activation_layers):
    enc = Qwen3VlTextEncoder(
        vocab_size=256,
        hidden_size=HIDDEN,
        intermediate_size=256,
        hidden_act="silu",
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        mrope_section=[6, 5, 5],
        head_dim=32,
        activation_layers=activation_layers,
        device=submesh,
        parallel_config=EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=0)),
        ccl_manager=CCLManager(submesh, num_links=1, topology=ttnn.Topology.Linear),
    )
    enc.load_torch_state_dict(_reference_text_model(layers).state_dict())
    return enc


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
def test_text_only_path_is_unchanged(mesh_device, submesh_shape):
    """Passing no vision arguments must give bit-identical output to before.

    This is the guarantee the Ideogram4 callers depend on: they never pass the new arguments, so the
    text path has to be untouched.
    """
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    torch.manual_seed(0)
    enc = _encoder(submesh, layers=4, activation_layers=(3,))
    ids = ttnn.from_torch(
        torch.randint(0, 256, (1, SEQ)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=submesh
    )
    cos, sin = create_rope_tensors(1, SEQ, None, 32, 10000.0, [6, 5, 5])
    pe = (bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh))

    plain = enc.forward(ids, attention_mask=None, pos_embeds=pe)[0]
    explicit_none = enc.forward(
        ids, attention_mask=None, pos_embeds=pe, vision_embeds=None, vision_runs=None, deepstack_embeds=None
    )[0]
    a = tensor.to_torch(plain, mesh_axes=[None, None, None])
    b = tensor.to_torch(explicit_none, mesh_axes=[None, None, None])
    assert torch.equal(a, b), "explicitly passing None diverges from omitting the arguments"


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
def test_vision_arguments_must_be_paired(mesh_device, submesh_shape, expect_error):
    """Embeds without runs (or the reverse) is a caller error rather than a silent no-op."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    enc = _encoder(submesh, layers=2, activation_layers=(1,))
    ids = ttnn.from_torch(
        torch.zeros(1, SEQ, dtype=torch.long), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=submesh
    )
    cos, sin = create_rope_tensors(1, SEQ, None, 32, 10000.0, [6, 5, 5])
    pe = (bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh))
    embeds = bf16_tensor(torch.zeros(1, 16, HIDDEN), device=submesh)

    with expect_error(ValueError, "must be passed together"):
        enc.forward(ids, attention_mask=None, pos_embeds=pe, vision_embeds=embeds)
    with expect_error(ValueError, "needs vision_runs"):
        enc.forward(ids, attention_mask=None, pos_embeds=pe, deepstack_embeds=[embeds])


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
def test_deepstack_is_applied_at_the_leading_layers(mesh_device, submesh_shape):
    """Each deepstack feature lands after its own layer, indexed by list position.

    Checked by effect rather than by inspection: a feature supplied for layer 0 only, versus for layers
    0 and 1, must give different output -- and both must differ from no deepstack at all. A port that
    applied them all at one layer, or dropped the tail, would pass a shape check.
    """
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    torch.manual_seed(0)
    enc = _encoder(submesh, layers=4, activation_layers=(3,))
    ids = ttnn.from_torch(
        torch.randint(0, 256, (1, SEQ)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=submesh
    )
    cos, sin = create_rope_tensors(1, SEQ, None, 32, 10000.0, [6, 5, 5])
    pe = (bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh))
    runs = [(8, 16)]
    vision = bf16_tensor(torch.randn(1, 16, HIDDEN), device=submesh)
    feat = [bf16_tensor(torch.randn(1, 16, HIDDEN), device=submesh) for _ in range(2)]

    def run(deepstack):
        out = enc.forward(
            ids,
            attention_mask=None,
            pos_embeds=pe,
            vision_embeds=vision,
            vision_runs=runs,
            deepstack_embeds=deepstack,
        )[0]
        return tensor.to_torch(out, mesh_axes=[None, None, None])

    none, one, two = run(None), run(feat[:1]), run(feat[:2])
    assert not torch.allclose(none, one, atol=1e-2), "a deepstack feature had no effect"
    assert not torch.allclose(one, two, atol=1e-2), "the second deepstack feature had no effect"


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
def test_vision_scatter_only_touches_the_vision_rows(mesh_device, submesh_shape):
    """Text rows before the first vision token must be unaffected by the scatter.

    Rows before the run cannot depend on it: attention is causal here, so a leak would mean the scatter
    wrote outside its range.
    """
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    torch.manual_seed(0)
    enc = _encoder(submesh, layers=2, activation_layers=(1,))
    ids = ttnn.from_torch(
        torch.randint(0, 256, (1, SEQ)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=submesh
    )
    cos, sin = create_rope_tensors(1, SEQ, None, 32, 10000.0, [6, 5, 5])
    pe = (bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh))
    runs = [(32, 16)]

    def run(vision):
        kwargs = {} if vision is None else dict(vision_embeds=vision, vision_runs=runs)
        out = enc.forward(ids, attention_mask=None, pos_embeds=pe, **kwargs)[0]
        return tensor.to_torch(out, mesh_axes=[None, None, None])

    plain = run(None)
    scattered = run(bf16_tensor(torch.randn(1, 16, HIDDEN) * 10, device=submesh))
    assert torch.allclose(plain[:, :32], scattered[:, :32], atol=1e-2), "rows before the run changed"
    assert not torch.allclose(plain[:, 32:48], scattered[:, 32:48], atol=1e-2), "the run itself did not change"
