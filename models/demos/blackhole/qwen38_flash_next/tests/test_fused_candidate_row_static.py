"""Static pins of the fused candidate row (``candidate_row``): the registry entry, the constants the kernels share with
the sampling chain, the scan's stable top-k insertion and the merge's core-order selection, the staging arithmetic, and
the call sites that feed the scan's shard row to the row's gather."""

from __future__ import annotations

import inspect
from pathlib import Path

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling_step as step
from models.demos.blackhole.qwen38_flash_next.ttnn import embedding, fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import greedy_tail as gt

KERNELS = Path(gt.__file__).parent / "kernels"


def test_registry_entry_is_bitwise_and_on_by_default():
    entry = fused.kernel(gt.CANDIDATE_ROW)
    assert entry.tolerance == fused.BITWISE and entry.default_on
    assert entry.fused is gt.sampling_candidates_fused and entry.composed is gt.sampling_candidates_chain
    assert fused.resolve(gt.CANDIDATE_ROW, {}) is gt.sampling_candidates_fused
    assert fused.resolve(gt.CANDIDATE_ROW, {fused.OFF_ENV: gt.CANDIDATE_ROW}) is gt.sampling_candidates_chain


def test_constants_match_the_sampling_chain():
    assert gt.CANDIDATES == embedding.SAMPLING_CANDIDATES_PER_DEVICE == 32
    assert gt.LIST_BYTES == 8 * gt.CANDIDATES
    assert 2 * gt.CANDIDATES * embedding.TP_SIZE == embedding.SAMPLING_CANDIDATE_ROW_SHAPE[3]
    assert gt.SCAN_ARGS[-2:] == ("lists_addr", "row") and gt.MERGE_ARGS[-3:] == (
        "lists_addr",
        "vocab_start_addr",
        "row_addr",
    )


def _one_line(source: str) -> str:
    return " ".join(source.split())  # clang-format wraps long statements; the pins read them as one line


def test_scan_keeps_a_stable_top_k_and_the_merge_picks_in_core_order():
    scan = _one_line((KERNELS / "scan.cpp").read_text())
    assert 'constexpr uint32_t CANDIDATES = get_named_compile_time_arg_val("candidates");' in scan
    assert "if (n == CANDIDATES && key <= ckey[CANDIDATES - 1])" in scan  # an earlier id keeps a boundary slot
    assert scan.count("#ifdef GT_CANDIDATE_ROW") == 1 and scan.count("#ifndef GT_CANDIDATE_ROW") == 1
    assert "while (j > 0 && ckey[j - 1] < key)" in scan  # moves left past strictly smaller keys only
    assert "list[CANDIDATES + i] = i < n ? cid[i] : 0xFFFFFFFFu" in scan
    assert "out[0] = static_cast<uint32_t>(cbits[0]) << 16;" in scan and "out[1] = cid[0];" in scan
    merge = _one_line((KERNELS / "merge.cpp").read_text())
    assert "static_assert(CANDIDATES == 0 || ROWS == 1" in merge
    assert merge.count("#ifdef GT_CANDIDATE_ROW") == 2
    host = inspect.getsource(gt.greedy_candidates)
    assert 'defines = [("GT_CANDIDATE_ROW", "1")] if candidates else []' in host and host.count("defines=defines") == 2
    assert "if (best_c == CORES || key > best_key)" in merge  # the first strict maximum over the heads in core order
    assert "gid.f = static_cast<float>(list_words[best_c * 2 * CANDIDATES + CANDIDATES + h] + start_id);" in merge
    assert "const uint32_t start_id = static_cast<uint32_t>(start.f);" in merge
    assert (
        "noc.async_write(stage, row, 8 * CANDIDATES, {.offset_bytes = STAGE_ROW}, {.page_id = 0, .offset_bytes = 0});"
        in merge
    )


def test_merge_staging_covers_the_kernel_layout():
    cores, rows, lanes, k = 40, 1, gt.PACKED_LANES_ROWS, gt.CANDIDATES
    pairs_bytes = ((16 * cores) + 63) & ~63
    stage_lists = ((2048 + rows * pairs_bytes + rows * 4 * lanes + 128) + 63) & ~63  # merge.cpp's STAGE_LISTS
    stage_row = stage_lists + cores * 8 * k + 64
    assert stage_row + 8 * k <= 2048 * gt.merge_stage_pages(rows, cores, lanes, k)
    assert gt.merge_stage_pages(rows, cores, lanes, 0) == gt.merge_stage_pages(rows, cores, lanes)
    assert gt.merge_stage_pages(1, cores, gt.PACKED_LANES) == gt.MERGE_STAGE_PAGES
    assert 128 * 49 + 16 + 8 * k <= 2048 * gt.SCAN_STAGE_PAGES  # 49 tiles per core at 40 cores


def test_the_scan_row_reaches_the_gather_at_every_call_site():
    epilogue = inspect.getsource(step.Qwen38SamplingChainExtension._epilogue)
    assert "self.lm_head.greedy_candidates(logits, candidate_row=self.constants)" in epilogue
    assert "self.lm_head.sampling_candidates(logits, self.constants, candidates=candidates)" in epilogue
    assert 'self.candidate_row = fused.enabled("candidate_row")' in inspect.getsource(
        step.Qwen38SamplingChainExtension.__init__
    )
    head = inspect.getsource(embedding.Qwen38TTNNLMHead.__init__)
    assert "fused_greedy_tail.sampling_candidates_fused" in head and "candidate_row folds into greedy_tail" in head
    fused_row = inspect.getsource(gt.sampling_candidates_fused)
    assert "ttnn.all_gather(" in fused_row and "ttnn.deallocate(candidates.shard_row)" in fused_row
    assert (
        "return type(lm_head).sampling_candidates(lm_head, logits, constants, into=into)" in fused_row
    )  # the chain fallback
