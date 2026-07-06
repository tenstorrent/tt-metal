from models.tt_dit.utils.ltx import apply_fast_env

# LTX_FAST=1 expands to the served fast-mode bundle at collection time — the pipeline reads the sigma
# schedule at import, before any fixture would run. apply_fast_env is the shared definition (see
# utils/ltx.py) so the pytest bundle can't drift from the ltx_server worker's.
apply_fast_env()
