# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Self-service performance optimization tool for TT hardware.

Subcommands (under `tt_hw_planner perf ...`):

  collect      Profile a brought-up model with Tracy + model_tracer.
  join         Cluster + classify a previously collected run.
  report       Write a self-contained Nsight-style HTML report.
  dashboard    Launch the interactive Dash app with live Apply buttons.
  blocks       List / show optimizer-block catalog.
  apply        Apply a named block to a cluster (writes a reversible patch).
  revert       Revert a previously applied block.
  compare      Diff two runs side-by-side.
  finalize     Convergence gate -> writes optimized_config.yaml + runner.sh.
"""

from .collect import collect_run, RunArtifacts  # noqa: F401
from .ceilings import BoxSpec, load_box_spec  # noqa: F401
