# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.pplx_decider_v1_27b.tt.decision import DecisionConfig
from models.demos.pplx_decider_v1_27b.tt.prefill_tuning import make_agmm_config_fn, make_rs_hparams, make_sdpa_config_fn
from models.demos.pplx_decider_v1_27b.tt.weight_mapping import PPLX_DECIDER_HF_MODEL, load_pplx_state_dict


class PplxDeciderModelArgs(Qwen36ModelArgs):
    def __init__(self, mesh_device=None, max_batch_size=1, max_seq_len=8192, **kwargs):
        # qwen36 defaults HF_MODEL to Qwen3.6-27B, so set ours first. The checkpoint has no mtp.* weights, so the
        # MTP head, which qwen36 builds for dense models by default, must be off.
        os.environ.setdefault("HF_MODEL", PPLX_DECIDER_HF_MODEL)
        super().__init__(
            mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, enable_mtp=False, **kwargs
        )
        self.decision_config = DecisionConfig.from_checkpoint(self.CKPT_DIR)
        # Prefill CCL optimisations default ON here (qwen36 keeps them off); an explicit env "0" turns one off.
        self.prefill_fused_dist_norm = os.environ.get("QWEN36_FUSED_DIST_NORM", "1") != "0"
        self.prefill_agmm_bf8 = os.environ.get("QWEN36_AGMM_BF8", "1") != "0"
        # Per-S prefill SDPA chunks / AGMM configs and the reduce-scatter hyperparameters from the op sweeps
        # (prefill_tuning.py); PPLX_SDPA_TUNED=0 / PPLX_AGMM_TUNED=0 / PPLX_RS_TUNED=0 turn one off.
        self.prefill_sdpa_config_fn = make_sdpa_config_fn(self.num_devices)
        self.prefill_agmm_config_fn = make_agmm_config_fn(self.num_devices)
        self.prefill_rs_hparams = make_rs_hparams()
        if self.num_devices > 1:
            logger.info(
                f"pplx prefill CCL: fused_dist_norm={self.prefill_fused_dist_norm} agmm_bf8={self.prefill_agmm_bf8} "
                f"ccl_topology={self.ccl_topology()} (num_devices={self.num_devices}, "
                f"fabric={ttnn.get_fabric_config()}) tuned sdpa={self.prefill_sdpa_config_fn is not None} "
                f"agmm={self.prefill_agmm_config_fn is not None} rs={self.prefill_rs_hparams}"
            )

    def ccl_topology(self):
        """CCL topology; PPLX_CCL_TOPOLOGY = auto (default) | base | ring | linear.

        auto: Ring when the mesh has > 2 devices and the active fabric is a ring/torus (FABRIC_1D_RING or
        FABRIC_2D_TORUS_*), else the base (tt_transformers) choice. base: always the base choice. ring: as auto but
        warns once when it has to fall back to base. linear: always Linear.
        """
        mode = os.environ.get("PPLX_CCL_TOPOLOGY", "auto").strip().lower() or "auto"
        if mode == "base":
            return super().ccl_topology()
        if mode == "linear":
            return ttnn.Topology.Linear
        if mode not in ("auto", "ring"):
            raise ValueError(f"PPLX_CCL_TOPOLOGY must be auto|base|ring|linear, got {mode!r}")
        ring_fabrics = (
            ttnn.FabricConfig.FABRIC_1D_RING,
            ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
            ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        )
        fabric = ttnn.get_fabric_config()
        if self.num_devices > 2 and fabric in ring_fabrics:
            return ttnn.Topology.Ring
        if mode == "ring" and not getattr(self, "_ccl_ring_warned", False):
            self._ccl_ring_warned = True
            logger.warning(
                f"PPLX_CCL_TOPOLOGY=ring ignored (num_devices={self.num_devices}, fabric={fabric}); using base topology"
            )
        return super().ccl_topology()

    def load_state_dict(self):
        return load_pplx_state_dict(self.CKPT_DIR, self.decision_config.token_ids, self.vocab_size, self.dim)
