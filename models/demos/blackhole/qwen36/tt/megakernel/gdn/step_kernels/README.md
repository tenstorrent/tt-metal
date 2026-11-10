gdn_spec_step kernels (compute / reader / writer) for the engine GDN role (G2), copied verbatim from
anatarajan/qwen38-r16-gdn-u1-onk @ bf5249feb24 (ttnn/cpp/ttnn/operations/experimental/kda/gdn_spec_step/device/kernels).
Every new path is behind a define (GDN_PREFIXED*, GDN_BATCH, GDN_RT_BM, GDN_LOOP, GDN_STATE_FIRST, GDN_IN_WAIT,
GDN_EXT_SEM, GDN_OUT_SIGNAL, GDN_ST_RING); the shared helpers (gdn_step_gate_math.hpp, gdn_step_dataflow_helpers.hpp, ...) are the
tree's stock ones. engine/gdn_role.py compiles them by absolute path (MKA_GDN_KDIR overrides).
