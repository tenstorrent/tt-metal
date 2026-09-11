import pytest

from models.common.sampling.tt_sampling import TTSampling


def test_qwen_tp4_local_topk_plan_uses_two_legal_multicore_widths():
    chunk_width, padded_width = TTSampling._plan_local_topk_chunks(62_080, 2)

    assert chunk_width == 32_768
    assert padded_width == 65_536
    assert chunk_width < 65_535


@pytest.mark.parametrize("num_chunks", [0, 3])
def test_local_topk_plan_rejects_invalid_chunk_counts(num_chunks, expect_error):
    with expect_error(ValueError, "local_topk_num_chunks must be a positive power of two"):
        TTSampling._plan_local_topk_chunks(62_080, num_chunks)


def test_tp4_forward_enters_requested_local_chunks(monkeypatch, expect_error):
    """The TP4 opt-in must not be gated behind single-device reduction."""
    import ttnn

    sampler = object.__new__(TTSampling)
    sampler._force_argmax_sampling = False
    sampler.multi_step_reduction = False
    sampler.local_topk_num_chunks = 2
    sampler.local_topk_chunk_width = 32768
    sampler.sub_core_grids = None
    sampler._mask_invalid_vocab_logits = lambda x: x
    monkeypatch.setattr(ttnn, "typecast", lambda x, **kwargs: x)

    class ReachedLocalChunks(Exception):
        pass

    def split(x):
        raise ReachedLocalChunks("requested local chunks")

    sampler._split_local_topk_logits = split
    with expect_error(ReachedLocalChunks, "requested local chunks"):
        sampler.forward(object())


def test_local_chunks_tile_before_padding_and_preserve_boundary_ids(monkeypatch):
    """CPU-backed op doubles check layout ordering, invalid padding, and IDs."""
    import torch

    import ttnn

    sampler = object.__new__(TTSampling)
    sampler.local_topk_chunk_width = 32768
    sampler.local_topk_padded_width = 65536
    sampler.sub_core_grids = None
    converted = []

    def to_layout(x, layout):
        assert layout == ttnn.TILE_LAYOUT
        converted.append(x)
        return x

    def pad(x, padding, *, value, **kwargs):
        assert converted and converted[-1] is x
        return torch.nn.functional.pad(x, (0, padding[-1][1]), value=value)

    monkeypatch.setattr(ttnn, "to_layout", to_layout)
    monkeypatch.setattr(ttnn, "pad", pad)
    monkeypatch.setattr(ttnn, "split", lambda x, width, dim: list(torch.split(x, width, dim=dim)))
    logits = torch.full((1, 1, 4, 62080), -100.0)
    expected = torch.tensor([0, 32767, 32768, 62079])
    logits[0, 0, torch.arange(4), expected] = -1.0
    chunks = sampler._split_local_topk_logits(logits)
    assert [x.shape[-1] for x in chunks] == [32768, 32768]
    assert torch.all(chunks[-1][..., 29312:] < -100.0)
    values, indices = zip(*(chunk.max(dim=-1) for chunk in chunks))
    values = torch.stack(values, dim=-1)
    indices = torch.stack([index + i * 32768 for i, index in enumerate(indices)], dim=-1)
    selected = indices.gather(-1, values.argmax(dim=-1, keepdim=True)).reshape(-1)
    assert torch.equal(selected, expected)


def test_local_chunks_reject_replicated_full_vocabulary(expect_error):
    from types import SimpleNamespace

    sampler = object.__new__(TTSampling)
    sampler.local_topk_padded_width = 65536
    with expect_error(ValueError, "preserve vocabulary sharding"):
        sampler._split_local_topk_logits(SimpleNamespace(shape=(1, 1, 32, 248064)))


def test_local_topk_plan_rejects_uint16_offset_wrap(expect_error):
    with expect_error(ValueError, "at most 65536 entries"):
        TTSampling._plan_local_topk_chunks(131072, 4)


@pytest.mark.parametrize(
    "cluster_shape,vocab_size,max_top_k,error",
    [
        ((1, 1), 62080, 32, "require multi-device vocabulary sampling"),
        ((1, 4), 128, 32, "chunk width must be at least max_top_k"),
    ],
)
def test_explicit_local_chunk_geometry_fails_before_upload(
    cluster_shape, vocab_size, max_top_k, error, monkeypatch, expect_error
):
    from types import SimpleNamespace

    import ttnn

    monkeypatch.setattr(ttnn.device, "is_wormhole_b0", lambda mesh: False)
    monkeypatch.setattr(ttnn.device, "is_blackhole", lambda mesh: True)
    args = SimpleNamespace(
        cluster_shape=cluster_shape,
        vocab_size=vocab_size,
        padded_vocab_size=vocab_size,
        local_topk_num_chunks=2,
        pad_logits_to_power_of_2=True,
        max_top_k=max_top_k,
    )
    with expect_error(ValueError, error):
        TTSampling(args=args, mesh_device=SimpleNamespace(shape=cluster_shape), tt_ccl=None)
