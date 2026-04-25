---
name: debug-kernel
description: Debug compilation or runtime errors in LLK kernels. Infers architecture and kernel type from path.
user_invocable: true
---

# /debug-kernel — Debug an LLK Kernel

## Usage

```
/debug-kernel tt_llk_blackhole/llk_lib/llk_pack_untilize.h
/debug-kernel tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_gelu.h
/debug-kernel  (no args — will ask for the kernel path or check recent test output)
```

## What to Do

1. Parse the kernel path from the user's arguments. If not provided, check recent test output or ask.
2. Infer the kernel type from the path:
   - `common/inc/sfpu/` → sfpu
   - `llk_lib/llk_math_` → math
   - `llk_lib/llk_pack_` → pack
   - `llk_lib/llk_unpack_` → unpack
3. Infer the architecture from the path:
   - `tt_llk_wormhole_b0/` → wormhole
   - `tt_llk_blackhole/` → blackhole
   - `tt_llk_quasar/` → quasar
4. Spawn the **llk-debugger** agent:

```
Agent tool:
  subagent_type: "llk-debugger"
  description: "Debug {kernel_name}"
  prompt: |
    Kernel path: {kernel_path}
    Kernel type: {kernel_type}
    Architecture: {architecture}
    Error context: {any error output the user provided, or "check logs"}
```

5. Report the agent's results to the user
6. If the fix involves code changes, show the diff and explain the fix
