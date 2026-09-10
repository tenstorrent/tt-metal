from models.tt_dit.utils.ltx import apply_quality_env

# LTX_QUALITY=high|medium|fast (or legacy LTX_FAST=1) expands to the served bundle at collection time
# — the pipeline reads the sigma schedule at import, before any fixture would run. apply_quality_env
# is the shared definition (see utils/ltx.py) so the pytest bundle can't drift from the ltx_server
# worker's; with no LTX_QUALITY set it falls back to apply_fast_env (LTX_FAST=1 still works).
apply_quality_env()
