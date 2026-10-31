# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks (no device opened by the tests) for the Qwen ring SDPA q_chunk picker and the ring
program config (SDPA_RING_PROGCFG) built from the real SDPA_PROGCFG of TtQwenModelArgs. Picks were measured
(ring SDPA micro-benchmark) only up to 40960; above that the picker returns the default.
Note: the directory conftest (ensure_devices_tg) enumerates the chips, so run with the devices idle."""

import pytest

from models.demos.llama3_70b_galaxy.tests.qwen_host_model_args import qwen_model_args
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import (
    RING_SDPA_LONG_K_CHUNK,
    RING_SDPA_MAX_MEASURED_SEQ_LEN,
    qwen_ring_sdpa_program_config,
    ring_sdpa_balanced_q_chunk,
    ring_sdpa_k_chunk,
)

pytestmark = pytest.mark.host_only  # no device: these tests run anywhere

N_LOCAL_HEADS = 8  # Qwen3-32B: 64 q heads / 8 kv heads per device row
NUM_CORES = 70  # 7 x 10 SDPA grid


@pytest.fixture(scope="module")
def qwen_cfg():
    # The real Qwen model config (TtQwenModelArgs.model_config): SDPA_PROGCFG and SDPA_RING_PROGCFG
    # (_sdpa_ring_progcfg in qwen_model_config.py), built on the host without a device.
    args = qwen_model_args(max_seq_len=40960)
    assert not args.is_blackhole, "the ring picker is Wormhole-only; run on the Wormhole Galaxy"
    return args


def _fields(cfg):
    grid = cfg.compute_with_storage_grid_size
    return (grid.x, grid.y), cfg.q_chunk_size, cfg.k_chunk_size, cfg.exp_approx_mode


@pytest.mark.parametrize(
    "seq_len, q_chunk",
    [
        (8192, 64),
        (16384, 128),
        (24576, 192),
        (32768, 256),
        (40960, 320),
        # above the measured range: default, although the busiest-core model would pick otherwise
        (49152, 256),
        (65536, 256),
        (81920, 256),
        (131072, 256),
    ],
)
def test_picker_buckets(seq_len, q_chunk):
    assert ring_sdpa_balanced_q_chunk(seq_len, n_local_heads=N_LOCAL_HEADS, num_cores=NUM_CORES) == q_chunk


def test_picker_default_above_measured_range():
    assert RING_SDPA_MAX_MEASURED_SEQ_LEN == 40960
    for seq_len in range(RING_SDPA_MAX_MEASURED_SEQ_LEN + 2048, 262145, 2048):
        assert ring_sdpa_balanced_q_chunk(seq_len, N_LOCAL_HEADS, NUM_CORES) == 256, seq_len


def test_picker_valid_for_ring_sdpa():
    # ring_distributed SDPA (ring_size 4) needs S / 8 divisible by q_chunk, for every length the picker sees.
    for seq_len in range(6144, RING_SDPA_MAX_MEASURED_SEQ_LEN + 1, 2048):
        q = ring_sdpa_balanced_q_chunk(seq_len, N_LOCAL_HEADS, NUM_CORES)
        assert (seq_len // 8) % q == 0, (seq_len, q)


@pytest.mark.parametrize("seq_len", [8192, 16384, 24576, 32768, 40960])
def test_ring_program_config_keeps_base_fields(qwen_cfg, seq_len):
    # SDPA_RING_PROGCFG above the traced buckets = the real SDPA_PROGCFG with q_chunk_size and k_chunk_size
    # replaced (k640 only where it divides S / 8, i.e. 40960) and the approximate exponential
    # (exp_approx_mode=True); SDPA_PROGCFG itself keeps the exact one.
    base = qwen_cfg.model_config["SDPA_PROGCFG"](seq_len, chunk_start_idx=0)
    ring = qwen_cfg.model_config["SDPA_RING_PROGCFG"](seq_len, chunk_start_idx=0)
    (bgrid, bq, bk, bexp), (rgrid, rq, rk, rexp) = _fields(base), _fields(ring)
    assert rgrid == bgrid
    assert rk == ring_sdpa_k_chunk(seq_len, bk)
    assert rk == (RING_SDPA_LONG_K_CHUNK if seq_len == 40960 else bk)
    assert rgrid == (7, 10)  # the grid the picker was measured with
    assert bexp is False and rexp is True
    n_local_heads = qwen_cfg.n_heads // qwen_cfg.n_kv_heads
    assert n_local_heads == N_LOCAL_HEADS
    assert rq == ring_sdpa_balanced_q_chunk(seq_len, n_local_heads, bgrid[0] * bgrid[1], default_q_chunk=bq)
    assert _fields(qwen_ring_sdpa_program_config(base, seq_len, n_local_heads, exp_approx_mode=True)) == _fields(ring)


@pytest.mark.parametrize(
    "seq_len, chunk_start_idx", [(1024, 0), (2048, 0), (4096, 0), (8192, 2048), (40960, 4096), (49152, 0)]
)
def test_ring_program_config_is_base_outside_picker(qwen_cfg, seq_len, chunk_start_idx):
    # <= 4096 (traced buckets), chunked prefill (chunk_start_idx > 0) and lengths above the 40960 cap
    # (QWEN_OPT_MAX_SEQ_LEN) keep the real SDPA_PROGCFG.
    base = qwen_cfg.model_config["SDPA_PROGCFG"](seq_len, chunk_start_idx=chunk_start_idx)
    ring = qwen_cfg.model_config["SDPA_RING_PROGCFG"](seq_len, chunk_start_idx=chunk_start_idx)
    assert _fields(ring) == _fields(base)


def test_ring_program_config_40960_measured_pick(qwen_cfg):
    # The measured keeps (02/02b, 03, 19): 40960 runs q320 / k640 / approximate exp instead of SDPA_PROGCFG's
    # q256 / k512 / exact exp.
    assert _fields(qwen_cfg.model_config["SDPA_RING_PROGCFG"](40960)) == ((7, 10), 320, 640, True)
    assert _fields(qwen_cfg.model_config["SDPA_PROGCFG"](40960)) == ((7, 10), 256, 512, False)


@pytest.mark.parametrize(
    "seq_len, q_chunk, k_chunk",
    [
        (16384, 128, 512),  # S / 8 = 2048: not a multiple of 640
        (24576, 192, 512),  # 3072
        (32768, 256, 512),  # 4096
        (40960, 320, 640),  # 5120 = 8 * 640
    ],
)
def test_ring_program_config_prefill_bucket_table(qwen_cfg, seq_len, q_chunk, k_chunk):
    ring = qwen_cfg.model_config["SDPA_RING_PROGCFG"](seq_len, chunk_start_idx=0)
    assert _fields(ring) == ((7, 10), q_chunk, k_chunk, True)


def test_ring_k_chunk_picker():
    assert ring_sdpa_k_chunk(40960, 512) == 640
    for seq_len in (8192, 16384, 24576, 32768, 49152, 81920):  # not measured / 640 does not divide S / 8
        assert ring_sdpa_k_chunk(seq_len, 512) == 512, seq_len
    # above the measured range the default is kept even where 640 divides the slice
    assert ring_sdpa_k_chunk(RING_SDPA_MAX_MEASURED_SEQ_LEN * 2, 512) == 512
