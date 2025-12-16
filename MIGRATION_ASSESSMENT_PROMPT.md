# Migration Assessment: ComfyUI-tt → ComfyUI-tt_standalone

## Objective
Assess whether any work done in `/home/tt-admin/ComfyUI-tt/` needs to be migrated to `/home/tt-admin/ComfyUI-tt_standalone/` before abandoning the old repository.

---

## Context

### Original Plan (Now Changing)
- Was modifying `/home/tt-admin/ComfyUI-tt/` (existing integration)
- Applied fixes to `custom_nodes/tenstorrent_nodes/wrappers.py`

### New Direction
- **Abandon**: `/home/tt-admin/ComfyUI-tt/`
- **Use**: `/home/tt-admin/ComfyUI-tt_standalone/` as the clean base
- **Bridge**: Will be built from scratch to connect standalone to TT hardware

### Question
Do any of our Phase 0 fixes need to be migrated? Or can we start fresh?

---

## Work Completed in ComfyUI-tt

### File Modified
**Location**: `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/wrappers.py`

**Changes Made**:

1. **Timestep Conversion Fix** (Line 363)
   ```python
   t_discrete = self.model_sampling.timestep(t).float()
   ```

2. **CFG Unbatching Fix** (Lines 231-357)
   - Refactored `apply_model()` to detect batch size 2
   - Created `_apply_single()` method
   - Splits `[uncond, cond]` → 2 separate calls → recombines

3. **Enhanced Logging** (Lines 296-304, 360-379)
   - Timestep format diagnostics
   - CFG batch detection
   - Bridge call tracking

### Dependencies
- Uses `TenstorrentBackend` class from `comfy/backends/tenstorrent_backend.py`
- Expects bridge server at `/tmp/tt-comfy.sock` (Unix socket)
- Uses shared memory tensor transfer via `TensorBridge`

---

## Investigation Tasks

### Task 1: Understand ComfyUI-tt_standalone Architecture

**Agent**: `explore` (thoroughness: "very thorough")

**Prompt**:
```
Explore the ComfyUI-tt_standalone repository to understand its architecture.

Location: /home/tt-admin/ComfyUI-tt_standalone/

Questions to answer:
1. Is this a clean, unmodified ComfyUI installation?
2. Does it have any Tenstorrent-specific modifications already?
3. What's in the custom_nodes directory (if any)?
4. Does it have a backends system like ComfyUI-tt?
5. What version of ComfyUI is this based on?

Key directories to examine:
- /home/tt-admin/ComfyUI-tt_standalone/comfy/
- /home/tt-admin/ComfyUI-tt_standalone/custom_nodes/
- /home/tt-admin/ComfyUI-tt_standalone/comfy/backends/ (if exists)

Deliverable:
- Architecture summary
- List of existing custom nodes/backends
- Comparison with ComfyUI-tt structure
```

---

### Task 2: Assess Applicability of Fixes to Standalone

**Agent**: `problem-investigator`

**Prompt**:
```
Analyze whether the fixes from ComfyUI-tt apply to ComfyUI-tt_standalone.

Context:
We made two fixes in ComfyUI-tt:
1. Timestep conversion (sigma → discrete timestep)
2. CFG unbatching (batch of 2 → 2 separate calls)

These were needed because:
- TTModelWrapper didn't inherit from ComfyUI's BaseModel
- TT pipeline expects separate uncond/cond calls, not batched

Questions to investigate:

1. **New Architecture Question**:
   - Are we building a NEW custom node in ComfyUI-tt_standalone?
   - Or modifying existing ComfyUI internals?
   - Or creating a backend system?

2. **Timestep Fix Relevance**:
   - If we build a custom node that inherits from BaseModel, do we need the fix?
   - Or will BaseModel's timestep conversion happen automatically?
   - Check: /home/tt-admin/ComfyUI-tt_standalone/comfy/model_base.py line 182

3. **CFG Fix Relevance**:
   - Does ComfyUI-tt_standalone's sampling code batch CFG?
   - Check: /home/tt-admin/ComfyUI-tt_standalone/comfy/samplers.py
   - Will we need to handle batched or separate calls?

4. **Bridge Architecture**:
   - If we're building a bridge from scratch, what protocol?
   - Will it use Unix socket like old approach?
   - Or different architecture (HTTP, direct Python import)?

Files to examine:
- /home/tt-admin/ComfyUI-tt_standalone/comfy/model_base.py
- /home/tt-admin/ComfyUI-tt_standalone/comfy/samplers.py
- /home/tt-admin/ComfyUI-tt_standalone/comfy/model_patcher.py

Deliverable:
- Assessment: "Fixes needed" or "Not needed" for each fix
- Rationale for each decision
- Recommended architecture for new integration
```

---

### Task 3: Identify Reusable Code

**Agent**: `local-file-searcher`

**Prompt**:
```
Search ComfyUI-tt for reusable code components that should be migrated.

Search locations:
- /home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/
- /home/tt-admin/ComfyUI-tt/comfy/backends/

Search for:
1. Utility functions that could be reused
   - Tensor validation (validate_latent_shape)
   - Tensor serialization (serialize_tensor, deserialize_tensor)
   - Configuration helpers (get_model_config)

2. Bridge communication code
   - TensorBridge class (shared memory management)
   - Socket communication protocol
   - Message packing/unpacking

3. Node implementations that worked
   - TT_FullDenoise (reportedly SSIM 0.998+)
   - TT_CheckpointLoader
   - Any proven working components

Search patterns:
- "class TensorBridge"
- "def validate_latent_shape"
- "def serialize_tensor"
- "class TT_FullDenoise"
- "class TTModelWrapper"

Deliverable:
- List of reusable components with file paths
- Assessment of each component's value
- Recommendation: migrate or rewrite?
```

---

### Task 4: Compare Standalone Server Integration Approaches

**Agent**: `code-writer` (research mode, no code changes yet)

**Prompt**:
```
Research and compare different approaches for integrating ComfyUI-tt_standalone with the standalone SDXL server.

Standalone server info:
- Location: /home/tt-admin/tt-metal/sdxl_server.py
- Architecture: HTTP REST API (FastAPI)
- Endpoints: /image/generations
- Working: Already tested, generates images successfully

Integration options to evaluate:

Option 1: Custom Node → HTTP Client
- ComfyUI custom node makes HTTP requests to standalone server
- Pro: Clean separation, no bridge needed
- Con: HTTP overhead, serialization

Option 2: Custom Node → Direct Python Import
- ComfyUI custom node imports and uses SDXLRunner directly
- Pro: No IPC overhead, fastest
- Con: Shared Python environment, potential conflicts

Option 3: Unix Socket Bridge (like old approach)
- Separate bridge process between ComfyUI and standalone
- Pro: Isolation, proven pattern
- Con: Extra process, shared memory complexity

Option 4: ComfyUI Backend System
- Implement backend like comfy/backends/tenstorrent_backend.py
- Pro: Clean ComfyUI integration pattern
- Con: More complex, needs backend API implementation

For each option, analyze:
1. Does timestep fix still apply?
2. Does CFG fix still apply?
3. Complexity (1-10)
4. Performance impact
5. Maintainability

Files to reference:
- /home/tt-admin/tt-metal/sdxl_server.py
- /home/tt-admin/tt-metal/sdxl_runner.py
- /home/tt-admin/ComfyUI-tt_standalone/comfy/model_base.py

Deliverable:
- Comparison matrix of 4 options
- Recommendation with rationale
- Assessment of whether our fixes are needed for recommended option
```

---

### Task 5: Create Migration Plan (If Needed)

**Agent**: `planner-agent`

**Prompt**:
```
Based on the findings from Tasks 1-4, create a migration plan.

Inputs (you'll receive these from previous tasks):
- ComfyUI-tt_standalone architecture
- Applicability assessment of fixes
- List of reusable components
- Recommended integration approach

Create a plan that includes:

1. **What to Migrate** (or "Nothing - start fresh")
   - List each component/fix
   - Why it's needed in new architecture
   - Where it should go in ComfyUI-tt_standalone

2. **What to Abandon**
   - Components that don't apply to new architecture
   - Rationale for each

3. **Implementation Steps**
   - If migration needed: Step-by-step migration plan
   - If starting fresh: Step-by-step build plan
   - Estimated time for each step

4. **Testing Strategy**
   - How to verify migrated code works
   - Comparison with old implementation

Deliverable format:
```markdown
# Migration Plan: ComfyUI-tt → ComfyUI-tt_standalone

## Executive Summary
[Migrate X components | Start completely fresh]

## Components to Migrate
1. [Component name]
   - Source: /home/tt-admin/ComfyUI-tt/path/to/file
   - Destination: /home/tt-admin/ComfyUI-tt_standalone/path/to/file
   - Modifications needed: [List]
   - Reason: [Why needed]

## Components to Abandon
1. [Component name]
   - Reason: [Why not needed in new architecture]

## Implementation Steps
[Detailed steps]

## Testing Plan
[How to verify]
```

---

## Execution Sequence

**Run these tasks in order using the Agent tool:**

1. **Explore** ComfyUI-tt_standalone architecture
   ↓
2. **Problem-Investigator** assesses fix applicability
   ↓
3. **Local-File-Searcher** finds reusable code
   ↓
4. **Code-Writer** (research) compares integration approaches
   ↓
5. **Planner-Agent** creates final migration plan
   ↓
6. **Integration-Orchestrator** coordinates implementation (if migration needed)

---

## Decision Points

### After Task 2:
**If fixes not needed in new architecture** → Skip migration, start fresh
**If fixes needed** → Continue to Task 3-5

### After Task 4:
**If recommended approach is "direct Python import"** → Fixes likely not needed (using full pipeline)
**If recommended approach is "custom node"** → Fixes likely needed (same issues as before)

---

## Success Criteria

By the end of this assessment, we should know:
1. ✅ Whether to migrate anything from ComfyUI-tt
2. ✅ Which integration approach to use for ComfyUI-tt_standalone
3. ✅ Whether our timestep/CFG fixes are still relevant
4. ✅ A clear implementation plan for the chosen approach

---

## Estimated Time

- Task 1 (Explore): 15-30 minutes
- Task 2 (Investigate): 30-45 minutes
- Task 3 (Search): 15-20 minutes
- Task 4 (Compare): 30-45 minutes
- Task 5 (Plan): 30-60 minutes

**Total**: 2-3 hours for complete assessment

---

## Deliverable

A comprehensive migration plan (or "start fresh" decision) with:
- Clear rationale
- Step-by-step implementation
- No wasted work migrating unnecessary components
- Optimal architecture for ComfyUI-tt_standalone integration

---

**Ready to execute?** Start with Task 1 using the `explore` agent.
