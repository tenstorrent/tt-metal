"""
Backward-compatible re-exports for ACE-Step Training V2 CLI.

Split structure:
    args.py           -- argparse construction (build_root_parser, build_fixed_standalone_parser, _add_* helpers)
    validation.py     -- path validation and target-module resolution
    config_builder.py -- LoRA/Training config construction from parsed args
    common.py         -- re-exports for API compatibility (this file)

Usage (unchanged)::

    from acestep.training_v2.cli.common import build_root_parser, build_configs
    from acestep.training_v2.cli.common import validate_paths, resolve_target_modules
    from acestep.training_v2.cli.common import build_fixed_standalone_parser
"""

from acestep.training_v2.cli.args import (  # noqa: F401
    _DEFAULT_NUM_WORKERS,
    VARIANT_DIR_MAP,
    build_fixed_standalone_parser,
    build_root_parser,
)
from acestep.training_v2.cli.config_builder import build_configs  # noqa: F401
from acestep.training_v2.cli.validation import resolve_target_modules, validate_paths  # noqa: F401
