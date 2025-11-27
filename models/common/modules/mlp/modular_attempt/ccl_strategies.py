# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CCL Strategies for MLP module.

Different hardware topologies require different collective communication patterns.
This module provides strategy implementations for each topology.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import ttnn
from models.common.modules.mlp.modular_attempt.mlp_config import HardwareTopology, MLPConfig


class CCLStrategy(ABC):
    """Base class for CCL strategies"""

    def __init__(self, mesh_device, tt_ccl, args):
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args

    @abstractmethod
    def reduce_after_ff1_ff3(
        self,
        w1_out: ttnn.Tensor,
        w3_out: ttnn.Tensor,
        mode: str,
        config: MLPConfig,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Reduce w1 and w3 outputs. Returns (w1_reduced, w3_reduced)"""

    @abstractmethod
    def all_gather_before_ff2(
        self,
        tensor: ttnn.Tensor,
        mode: str,
        config: MLPConfig,
        input_mem_cfg: Any,
    ) -> ttnn.Tensor:
        """All-gather before FF2 if needed. Returns tensor (possibly gathered)."""

    @abstractmethod
    def reduce_after_ff2(
        self,
        w2_out: ttnn.Tensor,
        mode: str,
        config: MLPConfig,
    ) -> ttnn.Tensor:
        """Reduce FF2 output. Returns reduced tensor."""


class SingleChipStrategy(CCLStrategy):
    """
    Strategy for single-chip systems (N150).
    No CCL operations needed - everything is local.
    """

    def reduce_after_ff1_ff3(self, w1_out, w3_out, mode, config):
        # No reduction needed on single chip
        return w1_out, w3_out

    def all_gather_before_ff2(self, tensor, mode, config, input_mem_cfg):
        # No gather needed
        return tensor

    def reduce_after_ff2(self, w2_out, mode, config):
        # No reduction needed
        return w2_out


class LinearTopologyStrategy(CCLStrategy):
    """
    Strategy for linear topology (N300, T3K).
    Uses reduce_scatter for all reductions.
    """

    def reduce_after_ff1_ff3(self, w1_out, w3_out, mode, config):
        # In linear topology, FF1/FF3 outputs don't need inter-device reduction
        # The weight sharding handles the parallelism
        return w1_out, w3_out

    def all_gather_before_ff2(self, tensor, mode, config, input_mem_cfg):
        # No gather needed in linear topology
        return tensor

    def reduce_after_ff2(self, w2_out, mode, config):
        """Reduce FF2 output using reduce_scatter + implicit all-gather"""
        from models.tt_transformers.tt.ccl import tt_all_reduce

        return tt_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            num_reduce_scatter_links=self.args.num_reduce_scatter_links,
            num_all_gather_links=self.args.num_all_gather_links,
            sharded=(mode == "decode"),
            memory_config=(
                config.memory_configs.ff2_out_reduce_scatter if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
            ),
            dtype=self.args.ccl_dtype,
            topology=self.args.ccl_topology(),
        )


class GalaxyTopologyStrategy(CCLStrategy):
    """
    Strategy for Galaxy (TG) 2D mesh topology.
    Uses different patterns based on model dimension.

    For dim >= 8192 or prefill: reduce_scatter + all_gather (composite)
    For dim < 8192 decode: all_reduce
    """

    def reduce_after_ff1_ff3(self, w1_out, w3_out, mode, config):
        """
        Reduce FF1/FF3 outputs across column axis.

        For large models (dim >= 8192) or prefill: use reduce_scatter
        Otherwise: use all_reduce
        """
        cluster_axis = 1

        if config.dim >= 8192 or mode == "prefill":
            return self._reduce_scatter_ff1_ff3(w1_out, w3_out, mode, config)
        else:
            return self._all_reduce_ff1_ff3(w1_out, w3_out, mode, config)

    def _reduce_scatter_ff1_ff3(self, w1_out, w3_out, mode, config):
        """Use reduce_scatter for FF1/FF3 (large models, prefill)"""
        cluster_axis = 1

        w1_reduced = ttnn.experimental.reduce_scatter_minimal_async(
            w1_out,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=self.args.num_reduce_scatter_links,
            cluster_axis=cluster_axis,
            memory_config=config.memory_configs.ff1_out_reduce_scatter if mode == "decode" else None,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        w3_reduced = ttnn.experimental.reduce_scatter_minimal_async(
            w3_out,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=1,
            cluster_axis=cluster_axis,
            memory_config=config.memory_configs.ff1_out_reduce_scatter if mode == "decode" else None,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        return w1_reduced, w3_reduced

    def _all_reduce_ff1_ff3(self, w1_out, w3_out, mode, config):
        """Use all_reduce for FF1/FF3 (small models decode)"""
        from models.tt_transformers.tt.ccl import tt_all_reduce

        w1_reduced = tt_all_reduce(
            w1_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            num_all_gather_links=2,
            sharded=(mode == "decode"),
            topology=self.args.ccl_topology(),
            memory_config=config.memory_configs.ff1_out_gathered if mode == "decode" else None,
        )

        w3_reduced = tt_all_reduce(
            w3_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            num_all_gather_links=2,
            sharded=(mode == "decode"),
            topology=self.args.ccl_topology(),
            memory_config=config.memory_configs.ff1_out_gathered if mode == "decode" else None,
        )

        return w1_reduced, w3_reduced

    def all_gather_before_ff2(self, tensor, mode, config, input_mem_cfg):
        """All-gather before FF2 for large models or prefill"""
        if config.dim < 8192 and mode != "prefill":
            # Small model decode - no gather needed (already done in all_reduce)
            return tensor

        cluster_axis = 1
        gathered = ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=2,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            memory_config=input_mem_cfg,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        if mode == "decode":
            gathered = ttnn.to_memory_config(gathered, ttnn.L1_MEMORY_CONFIG)

        return gathered

    def reduce_after_ff2(self, w2_out, mode, config):
        """Reduce FF2 output across row axis"""
        from models.tt_transformers.tt.ccl import tt_all_reduce

        return tt_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=0 if config.dim < 8192 else 3,
            num_reduce_scatter_links=self.args.num_reduce_scatter_links,
            num_all_gather_links=self.args.num_all_gather_links,
            sharded=(mode == "decode"),
            memory_config=(
                config.memory_configs.ff2_out_reduce_scatter if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
            ),
            dtype=self.args.ccl_dtype,
            use_composite=(config.dim >= 8192),
            topology=self.args.ccl_topology(),
        )


def create_ccl_strategy(
    topology: HardwareTopology,
    mesh_device,
    tt_ccl,
    args,
) -> CCLStrategy:
    """Factory function to create the appropriate CCL strategy"""

    strategies = {
        HardwareTopology.SINGLE_CHIP: SingleChipStrategy,
        HardwareTopology.LINEAR_1D: LinearTopologyStrategy,
        HardwareTopology.GALAXY_2D: GalaxyTopologyStrategy,
    }

    strategy_class = strategies.get(topology, SingleChipStrategy)
    return strategy_class(mesh_device, tt_ccl, args)
