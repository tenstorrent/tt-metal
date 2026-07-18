# Shard advisor status

The local `ttnn-advise` flow was invoked for the Qwen2.5-Coder-32B optimized
decoder before final TP4 tuning. After using the repository's expected Python
path, import failed while loading `libTTMLIRRuntime.so`: the shared library has
an undefined `moe_compute` symbol in this checkout/environment. No advisor
candidate was silently substituted.

The completed single-chip advisor artifact for
`advisor_packed_bfp8_hifi2_1d` remains the optimized baseline. For this stage,
the source compiler's TP=4 ownership plus direct Blackhole measurements of
precision, core grids, persistent CCL, payload dtype, fused collectives, and
boundary topologies are authoritative. The failure is tooling/environmental,
not a device or decoder runtime failure.
