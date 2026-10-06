# SPDX-License-Identifier: Apache-2.0
"""Register the missing HF architecture, then run the unchanged readiness CLI."""
import runpy
import sys

import torch

from . import hf_model  # explicit architecture registration

torch.set_num_threads(8)
runner = sys.argv.pop(1)
if runner not in ("generate", "run_prefill_check", "run_teacher_forcing", "run_autoregressive"):
    raise ValueError(runner)
runpy.run_module("readiness_check." + runner, run_name="__main__")
