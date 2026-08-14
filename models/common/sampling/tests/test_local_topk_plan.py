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
