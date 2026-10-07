# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real pinned plugin's serialized DP placement without devices."""

import os
from types import SimpleNamespace

import vllm
from vllm_tt_plugin.platform import TTPlatform, _load_standard_dp_mesh_grids, _load_standard_dp_visible_groups
from vllm_tt_plugin.worker import _bind_visible_devices_env, _resolve_mesh_grid

from models.demos.qwen38_27b_qb2.demo.galaxy_serving import additional_config, server_command


def test_pinned_plugin_binds_eight_disjoint_synthetic_groups(monkeypatch):
    assert vllm.__version__.startswith("0.26.0")
    groups = [",".join(str(i) for i in range(rank * 4, rank * 4 + 4)) for rank in range(8)]
    config = SimpleNamespace(additional_config=additional_config(groups))
    assert _load_standard_dp_visible_groups(config) == groups
    grids = _load_standard_dp_mesh_grids(config)
    assert grids == {group: (1, 4) for group in groups}
    monkeypatch.setattr(TTPlatform, "_standard_dp_mesh_grids", grids)
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "999")
    for rank, group in enumerate(groups):
        config.parallel_config = SimpleNamespace(
            data_parallel_rank_local=rank,
            data_parallel_index=rank,
            assigned_physical_gpu_ids=[int(chip) for chip in group.split(",")],
        )
        _bind_visible_devices_env(config)
        assert os.environ["TT_VISIBLE_DEVICES"] == group
        assert _resolve_mesh_grid("(8, 4)", 4, group) == (1, 4)


def test_pinned_plugin_rejects_conflicting_worker_assignment(expect_error):
    groups = [",".join(str(i) for i in range(rank * 4, rank * 4 + 4)) for rank in range(8)]
    config = SimpleNamespace(
        additional_config=additional_config(groups),
        parallel_config=SimpleNamespace(
            data_parallel_rank_local=0, data_parallel_index=0, assigned_physical_gpu_ids=[4, 5, 6, 7]
        ),
    )
    with expect_error(RuntimeError, "conflicts with discovery"):
        _bind_visible_devices_env(config)


def test_real_vllm_parser_accepts_galaxy_launch_arguments():
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    groups = [",".join(str(i) for i in range(rank * 4, rank * 4 + 4)) for rank in range(8)]
    command = server_command("/task", "/weights", groups)
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(command[4:])
    assert args.data_parallel_size == 8 and args.max_num_seqs == 16
    assert args.max_model_len == args.max_num_batched_tokens == 262144
    assert args.additional_config == additional_config(groups)
    assert args.enable_prefix_caching is False and args.enable_chunked_prefill is False
