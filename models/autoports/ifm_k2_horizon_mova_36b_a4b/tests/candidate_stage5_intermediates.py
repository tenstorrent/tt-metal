# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    def _save(self, name, value):
        if getattr(self, "debug_enabled", False):
            self.debug_tensors[name] = (
                value if name.startswith("rs_output:") else ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            )
        return value

    def _prefill_chunk(self, *args, **kwargs):
        self.debug_tensors = {}
        self.debug_enabled = True
        try:
            return super()._prefill_chunk(*args, **kwargs)
        finally:
            self.debug_enabled = False

    def _norm(self, x, name):
        return self._save("norm:" + name, super()._norm(x, name))

    def _linear(self, x, name):
        return self._save("linear:" + name, super()._linear(x, name))

    def _reduce(self, x, slot=0):
        self.debug_slot = slot
        self._save("reduce_input:" + str(slot), x)
        return self._save("reduce_output:" + str(slot), super()._reduce(x, slot))

    def _finish(self, residual, gate_logits, attention, **kwargs):
        self._save("attention", attention)
        return super()._finish(residual, gate_logits, attention, **kwargs)

    def _gather(self, x):
        self._save("rs_output:" + str(getattr(self, "debug_slot", -1)), x)
        return super()._gather(x)
