# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from models.ttt_compat.tt.lm_head import *  # noqa: F401,F403  # isort: skip
import models.ttt_compat.tt.lm_head as _tt
import ttnn
from models.demos.llama31_8b_p100.tt import wiring
from models.ttt_compat.tt.ccl import tt_all_reduce
from models.ttt_compat.tt.common import Mode


class LMHead(_tt.LMHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._matmul_bh = wiring.on("matmul_bh", self.args)

    def forward(self, x: ttnn.Tensor, debug_input_torch=None, debug_weight_torch=None):
        outputs = []
        use_prefetcher = self.prefetcher is not None and self.prefetcher.mode == Mode.DECODE
        split_sizes = self.split_sizes_ring_mm if use_prefetcher else self.split_sizes_dram_sharded
        program_configs = [
            self.args.get_lm_head_program_config(split_size, self.prefetcher if use_prefetcher else None)
            for split_size in split_sizes
        ]

        output_weights = self.output_weights_ring_mm if use_prefetcher else self.output_weights_dram_sharded

        if self.use_galaxy_no_prefetch:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        self.lm_head_output_memory_config = self.args.get_lm_head_output_mem_config(
            Mode.DECODE if use_prefetcher else Mode.PREFILL, self.prefetcher if use_prefetcher else None
        )

        for i, (weight, pc) in enumerate(zip(output_weights, program_configs)):
            output = wiring.linear(
                self._matmul_bh,
                x,
                weight,
                compute_kernel_config=self.compute_kernel_config,
                program_config=None if self.use_galaxy_no_prefetch else pc,
                memory_config=(
                    ttnn.DRAM_MEMORY_CONFIG if self.use_galaxy_no_prefetch else self.lm_head_output_memory_config
                ),
                dtype=self.args.lm_head_dtype if hasattr(self.args, "lm_head_dtype") else ttnn.bfloat8_b,
                sub_device_id=self.prefetcher.worker_sub_device_id if use_prefetcher else None,
            )
            output = ttnn.to_memory_config(
                output,
                memory_config=self.args.get_lm_head_sharded_output_mem_config(
                    self.prefetcher if use_prefetcher else None
                ),
            )

            outputs.append(output)

        ttnn.deallocate(x)

        output = ttnn.concat(
            outputs,
            dim=-1,
            memory_config=(
                self.args.model_config.get(
                    "LM_HEAD_OUTPUT_MEMCFG",
                    ttnn.DRAM_MEMORY_CONFIG
                    if self.padded_vocab_size // self.num_devices > 131072
                    else ttnn.L1_MEMORY_CONFIG,
                )
                if not use_prefetcher
                else ttnn.DRAM_MEMORY_CONFIG
            ),
            sub_core_grids=self.prefetcher.all_worker_cores_range_set if use_prefetcher else None,
        )

        if use_prefetcher:
            output = ttnn.to_memory_config(
                output,
                memory_config=self.args.get_lm_head_reshard_mem_config(self.prefetcher),
            )

        output = tt_all_reduce(
            output,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            dim=3 if self.args.is_galaxy else 0,
            memory_config=output.memory_config(),
            dtype=self.args.ccl_dtype,
            sharded=False,
            use_composite=True,
            subdevice_id=self.prefetcher.worker_sub_device_id if use_prefetcher else None,
        )

        return output
