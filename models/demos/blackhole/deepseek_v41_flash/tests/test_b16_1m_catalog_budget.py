# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU: the stack5 B=16 / 1M catalog entry (tools/mk_b16_1m_s5_spec.py) and the per-bank DRAM budget it relies on (dsv4-logs/b16_1m_plan/PLAN.md).

The anchors are MEMREPORT "end of the build" numbers of real servers (MiB per DRAM bank, fp8 pool, 1M, trace region 1.2e9):
  B=8 stack5 (wf_grid/on_34_v3): allocated 3744.2, free 187.2;  B=16 cand (b16_1m_r2/srv/r2, booted and served): allocated 3853.6, free 77.8.
"""
import copy
import importlib.util
import os

import pytest

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "mk_b16_1m_s5_spec.py")
spec = importlib.util.spec_from_file_location("mk_b16_1m_s5_spec", TOOL)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

PAGE_TOKENS = 128
PAGE_ROWS = 320  # rows of one 128-token page of the shared pool (sources 2, 8, 14: 64 rows each, source 20: 128)
ROW_BYTES_FP8 = 512
BANKS = 8
MIB = 1 << 20
FREE_B8_STACK5 = 187.2
FREE_B16_CAND = 77.8
ALLOC_B8_CAND = 3516.6
ALLOC_B16_CAND = 3853.6
APC_PAGED = 65.0  # stores, B=8
DUP_WEIGHTS = {"attn_front": 14e6, "skinny_wo": 10e6, "skinny_shared": 4.7e6}  # bytes per chip and layer
LAYERS = 40


def pool_pages(users_per_row, max_len=1048576, chunk=512):
    max_len = -(-max_len // chunk) * chunk
    return users_per_row * (-(-(max_len + PAGE_TOKENS) // PAGE_TOKENS))


def pool_mib_per_bank(users_per_row):
    return pool_pages(users_per_row) * PAGE_ROWS * ROW_BYTES_FP8 / BANKS / MIB


def catalog():
    env = {
        "DSV41_APC": "1",
        "DSV41_APC_IMAGES": "1",
        "DSV41_APC_GRID": "512",
        "DSV41_APC_PAGED": "1",
        "DSV41_APC_PAGED_XROW": "1",
        "DSV41_ATTN_FRONT": "1",
        "DSV41_ATTN_FRONT_ARM": "1",
        "DSV41_ATTN_FRONT_XSPLIT": "1",
        "DSV41_ATTN_FRONT_RING": "320",
        "DSV41_SKINNY_ARM": "1",
        "DSV41_SKINNY_WOAB": "1",
        "DSV41_SKINNY_SHARED": "1",
        "DSV41_SKINNY_CT": "2",
        "DSV41_FUSE_TAIL": "1",
        "DSV41_MHC_PRE16": "1",
        "DSV41_TRACE_REGION": "1200000000",
        "DSV41_DRAFT_MHC_PRE_MAX": "16",
    }
    entry = {
        "model_id": "id_deepseek-v41-flash-b8-1m_DeepSeek-V4.1-Flash_blackhole_galaxy",
        "impl": {"impl_id": "deepseek_v41_flash_b8_1m", "impl_name": "deepseek-v41-flash-b8-1m"},
        "env_vars": dict(env),
        "device_model_spec": {
            "default_impl": True,
            "max_concurrency": 8,
            "max_context": 1048576,
            "max_tokens_all_users_override": 8388608,
            "vllm_args": {
                "max_num_seqs": "8",
                "max_model_len": "1048576",
                "additional_config": '{"tt": {"l1_small_size": 16384, "trace_region_size": 1200000000}}',
            },
            "override_tt_config": {"l1_small_size": 16384, "trace_region_size": 1200000000},
            "env_vars": dict(env),
        },
    }
    return {"model_specs": {"deepseek-ai/DeepSeek-V4.1-Flash": {"BLACKHOLE_GALAXY": {"vLLM": {M.SRC_IMPL: entry}}}}}


def test_pool_pages_match_booted_servers():
    assert pool_pages(4) == 32772  # b16_1m_r2/srv/r2: "pool 32772 pages/row"
    assert pool_pages(2) == 16386  # B=8 cand; stack5 shows 16390 = +4 idle APC pages
    assert pool_mib_per_bank(1) == pytest.approx(160.0, abs=0.1)  # one user at 1M: 160 MiB/bank (common.py docstring)


def test_pool_pages_match_default_page_params():
    try:
        from models.demos.blackhole.deepseek_v41_flash.tt.common import default_page_params
    except Exception as e:  # ttnn not importable on a CPU-only host
        pytest.skip(f"tt.common needs ttnn: {e}")

    assert default_page_params(1048576, 4)["page_max_num_blocks_per_dp"] == pool_pages(4)


def test_cand_delta_is_the_pool():
    delta = ALLOC_B16_CAND - ALLOC_B8_CAND
    extra_pool = pool_mib_per_bank(4) - pool_mib_per_bank(2)
    assert 320.0 < delta < 345.0
    assert extra_pool == pytest.approx(320.0, abs=0.2)
    assert delta - extra_pool < 25.0  # rings, per-user buffers, key slabs


def test_stack5_as_is_does_not_fit_and_lean_does():
    pool_delta = pool_mib_per_bank(4) - pool_mib_per_bank(2) + 17.0
    as_is = FREE_B8_STACK5 - pool_delta
    assert as_is < 0  # the stock B=8 env at B=16 runs out of DRAM at boot
    dup = sum(DUP_WEIGHTS.values()) * LAYERS / BANKS / MIB
    assert 135 < dup < 150
    lean = FREE_B8_STACK5 + APC_PAGED + dup - pool_delta
    assert 45 < lean < 75  # PLAN.md: ~58 predicted, 77.8 measured on the pre-stack5 build
    assert lean < FREE_B16_CAND


def test_entry_is_b16_not_default_and_b8_untouched():
    full = catalog()
    before = copy.deepcopy(
        full["model_specs"]["deepseek-ai/DeepSeek-V4.1-Flash"]["BLACKHOLE_GALAXY"]["vLLM"][M.SRC_IMPL]
    )
    M.add_b16_entry(full)
    v = full["model_specs"]["deepseek-ai/DeepSeek-V4.1-Flash"]["BLACKHOLE_GALAXY"]["vLLM"]
    assert v[M.SRC_IMPL] == before
    e = v[M.NEW_IMPL]
    d = e["device_model_spec"]
    assert e["impl"]["impl_id"] == "deepseek_v41_flash_b16_1m" and e["impl"]["impl_name"] == "deepseek-v41-flash-b16-1m"
    assert "b16-1m" in e["model_id"] and "b8" not in e["model_id"]
    assert d["default_impl"] is False
    assert d["max_concurrency"] == 16 and d["max_context"] == 1048576
    assert d["max_tokens_all_users_override"] == 16 * 1048576
    assert d["vllm_args"]["max_num_seqs"] == "16" and d["vllm_args"]["max_model_len"] == "1048576"


def test_dead_weight_switches_off_and_booted_settings_kept():
    full = M.add_b16_entry(catalog())
    e = full["model_specs"]["deepseek-ai/DeepSeek-V4.1-Flash"]["BLACKHOLE_GALAXY"]["vLLM"][M.NEW_IMPL]
    for env in (e["env_vars"], e["device_model_spec"]["env_vars"]):
        for k in M.B16_OFF:
            assert env[k] == "0", k
        for k in M.B16_DROP:
            assert k not in env, k
        assert env["DSV41_MHC_PRE24"] == "0"
        assert env["DSV41_MHC_PRE16"] == "1"
        assert env["DSV41_TRACE_REGION"] == "1200000000"  # as booted at B=16 (82 of 143 MiB/bank used)
        assert env["DSV41_SPEC_MAX_CTX"] == str(1048576 + 256)
        assert env["MAX_NUM_SEQS"] == "16" and env["MAX_MODEL_LENGTH"] == "1048576"
        assert env["DSV41_APC"] == "1"  # host-tier APC stays on


def test_extra_override_wins():
    full = M.add_b16_entry(catalog(), {"DSV41_TRACE_REGION": "1000000000", "DSV41_SKINNY_WOAB": "1"})
    e = full["model_specs"]["deepseek-ai/DeepSeek-V4.1-Flash"]["BLACKHOLE_GALAXY"]["vLLM"][M.NEW_IMPL]
    assert e["env_vars"]["DSV41_TRACE_REGION"] == "1000000000" and e["env_vars"]["DSV41_SKINNY_WOAB"] == "1"


def test_real_stack5_catalog_if_present():
    import json

    p = "/mnt/tt-data/ssinghal/dsv4-logs/rc5/model_spec_rc5c_stack5.json"
    if not os.path.exists(p):
        pytest.skip("stack5 catalog not mounted")
    full = M.add_b16_entry(json.load(open(p)))
    v = full["model_specs"]["deepseek-ai/DeepSeek-V4.1-Flash"]["BLACKHOLE_GALAXY"]["vLLM"]
    assert sum(1 for x in v.values() if x["device_model_spec"]["default_impl"]) == 1
    assert v[M.SRC_IMPL]["device_model_spec"]["default_impl"] is True
    assert v[M.NEW_IMPL]["env_vars"]["DSV41_APC_PAGED"] == "0"


def test_allon_entry_keeps_every_optimization_and_adds_the_dram_levers():
    full = M.add_b16_entry(catalog(), all_on=True)
    e = full["model_specs"]["deepseek-ai/DeepSeek-V4.1-Flash"]["BLACKHOLE_GALAXY"]["vLLM"][M.NEW_IMPL_ALLON]
    assert e["impl"]["impl_id"] == "deepseek_v41_flash_b16_1m_allon" and e["device_model_spec"]["default_impl"] is False
    for env in (e["env_vars"], e["device_model_spec"]["env_vars"]):
        for k in (
            "DSV41_ATTN_FRONT",
            "DSV41_SKINNY_ARM",
            "DSV41_SKINNY_WOAB",
            "DSV41_SKINNY_SHARED",
            "DSV41_FUSE_TAIL",
            "DSV41_ATTN_FRONT_RING",
        ):
            assert env[k] != "0"  # all B=8 optimizations stay on
        assert (
            env["DSV41_COPY_GUARD"] == "1"
            and env["DSV41_EMBED_SHARD"] == "1"
            and env["DSV41_TRACE_REGION"] == "1000000000"
        )
        assert env["DSV41_APC_PAGED"] == "0" and env["DSV41_MHC_PRE24"] == "0" and env["MAX_NUM_SEQS"] == "16"


def test_allon_trace_region_reaches_the_device_config():
    import json

    full = M.add_b16_entry(catalog(), all_on=True)
    d = full["model_specs"]["deepseek-ai/DeepSeek-V4.1-Flash"]["BLACKHOLE_GALAXY"]["vLLM"][M.NEW_IMPL_ALLON][
        "device_model_spec"
    ]
    assert json.loads(d["vllm_args"]["additional_config"])["tt"]["trace_region_size"] == 1000000000
    assert d["override_tt_config"]["trace_region_size"] == 1000000000
