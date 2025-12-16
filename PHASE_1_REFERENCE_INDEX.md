# Phase 1 Reference Index

**Purpose:** Comprehensive index of all documents, code locations, and resources
**Last Updated:** December 16, 2025

---

## Document Index

### Primary Implementation Documents

| Document | Purpose | Path |
|----------|---------|------|
| Master Prompt | Complete implementation specification | `/home/tt-admin/tt-metal/PHASE_1_BRIDGE_EXTENSION_PROMPT.md` |
| Deployment Manifest | All artifacts and deployment info | `/home/tt-admin/tt-metal/PHASE_1_DEPLOYMENT_MANIFEST.md` |

### Execution Checklists

| Document | Purpose | Path |
|----------|---------|------|
| Phase 0 Checklist | Day-by-day validation tasks | `/home/tt-admin/tt-metal/PHASE_0_EXECUTION_CHECKLIST.md` |
| Week 1 Tasks | Detailed daily breakdown | `/home/tt-admin/tt-metal/WEEK_1_DETAILED_TASKS.md` |
| Week 2 Tasks | Detailed daily breakdown | `/home/tt-admin/tt-metal/WEEK_2_DETAILED_TASKS.md` |
| Week 3 Tasks | Detailed daily breakdown | `/home/tt-admin/tt-metal/WEEK_3_DETAILED_TASKS.md` |
| Week 4 Tasks | Detailed daily breakdown | `/home/tt-admin/tt-metal/WEEK_4_DETAILED_TASKS.md` |
| Week 5 Tasks | Detailed daily breakdown | `/home/tt-admin/tt-metal/WEEK_5_DETAILED_TASKS.md` |

### Frameworks and Guides

| Document | Purpose | Path |
|----------|---------|------|
| Success Criteria | Measurement methods | `/home/tt-admin/tt-metal/SUCCESS_CRITERIA_FRAMEWORK.md` |
| Risk Playbook | Risk response procedures | `/home/tt-admin/tt-metal/RISK_MITIGATION_PLAYBOOK.md` |
| Phase 0 Validation Guide | Detailed validation steps | `/home/tt-admin/tt-metal/PHASE_0_VALIDATION_GUIDE.md` |
| Architecture Context | Design rationale | `/home/tt-admin/tt-metal/ARCHITECTURE_CONTEXT.md` |

### Templates

| Document | Purpose | Path |
|----------|---------|------|
| Daily Standup | Daily team sync | `/home/tt-admin/tt-metal/DAILY_STANDUP_TEMPLATE.md` |
| Weekly Review | Weekly progress review | `/home/tt-admin/tt-metal/WEEKLY_REVIEW_CHECKLIST.md` |
| Code Review | Code review process | `/home/tt-admin/tt-metal/CODE_REVIEW_CHECKLIST.md` |
| Progress Dashboard | Metrics tracking | `/home/tt-admin/tt-metal/PROGRESS_DASHBOARD_TEMPLATE.md` |
| Weekly Status Report | Status communication | `/home/tt-admin/tt-metal/WEEKLY_STATUS_TEMPLATE.md` |
| Issue Escalation | Escalation process | `/home/tt-admin/tt-metal/ISSUE_ESCALATION_TEMPLATE.md` |
| Decision Log | Decision tracking | `/home/tt-admin/tt-metal/DECISION_LOG_TEMPLATE.md` |
| Reusability Comments | Code annotation format | `/home/tt-admin/tt-metal/REUSABILITY_COMMENT_TEMPLATE.md` |

### Decision Frameworks

| Document | Purpose | Path |
|----------|---------|------|
| Phase 0 Go/No-Go | Gate decision criteria | `/home/tt-admin/tt-metal/PHASE_0_GO_NO_GO_FRAMEWORK.md` |
| Scheduler Options | Scheduler design options | `/home/tt-admin/tt-metal/SCHEDULER_STATE_OPTIONS.md` |
| ControlNet Scenarios | ControlNet feasibility paths | `/home/tt-admin/tt-metal/CONTROLNET_FEASIBILITY_SCENARIOS.md` |

### Measurement Tools

| Document | Purpose | Path |
|----------|---------|------|
| Test Execution Matrix | Test schedule and expectations | `/home/tt-admin/tt-metal/TEST_EXECUTION_MATRIX.md` |
| Performance Budget Tracker | Latency tracking | `/home/tt-admin/tt-metal/PERFORMANCE_BUDGET_TRACKER.md` |

### Onboarding

| Document | Purpose | Path |
|----------|---------|------|
| Team Onboarding | New member orientation | `/home/tt-admin/tt-metal/TEAM_ONBOARDING.md` |

### Strategic Context

| Document | Purpose | Path |
|----------|---------|------|
| Parity Status Correction | Current state analysis | `/home/tt-admin/tt-metal/DEC15_PARITY_STATUS_CORRECTION.md` |
| Strategic Path Analysis | Path selection rationale | `/home/tt-admin/tt-metal/STRATEGIC_PATH_ANALYSIS.md` |
| Bridge to Integration Analysis | Native pathway details | `/home/tt-admin/tt-metal/BRIDGE_TO_INTEGRATION_ANALYSIS.md` |

---

## Code Locations

### Bridge Implementation

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py` | Operation handlers | `handle_denoise_step_single`, `handle_session_*` |
| `/home/tt-admin/tt-metal/comfyui_bridge/server.py` | Server entry point | `BridgeServer` |
| `/home/tt-admin/tt-metal/comfyui_bridge/model_config.py` | Model configurations | `MODEL_CONFIGS` |
| `/home/tt-admin/tt-metal/comfyui_bridge/session_manager.py` | Session management | `SessionManager`, `DenoiseSession` |

### ComfyUI Custom Nodes

| File | Purpose | Key Classes |
|------|---------|-------------|
| `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/nodes.py` | Node definitions | `TT_KSampler`, `TT_ApplyControlNet` |
| `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/__init__.py` | Node registration | `NODE_CLASS_MAPPINGS` |

### Test Files

| File | Purpose | Key Tests |
|------|---------|-----------|
| `/home/tt-admin/tt-metal/comfyui_bridge/tests/test_per_step.py` | Per-step API tests | `test_denoise_step_single_*` |
| `/home/tt-admin/tt-metal/comfyui_bridge/tests/test_session.py` | Session tests | `test_session_*` |
| `/home/tt-admin/tt-metal/comfyui_bridge/tests/test_controlnet.py` | ControlNet tests | `test_controlnet_*` |
| `/home/tt-admin/tt-metal/comfyui_bridge/tests/benchmark_per_step.py` | Performance benchmark | `benchmark_latency_overhead` |
| `/home/tt-admin/tt-metal/comfyui_bridge/tests/benchmark_ipc.py` | IPC latency benchmark | `measure_ipc_latency` |
| `/home/tt-admin/tt-metal/comfyui_bridge/tests/stress_test.py` | Stress testing | `stress_test` |
| `/home/tt-admin/tt-metal/comfyui_bridge/tests/validate_ssim.py` | SSIM validation | `compare_per_step_vs_full_loop` |

### Documentation Files (To Be Created)

| File | Purpose | Create In |
|------|---------|-----------|
| `/home/tt-admin/tt-metal/docs/architecture/scheduler_sync_design.md` | Scheduler sync design | Phase 0 Day 1-2 |
| `/home/tt-admin/tt-metal/docs/PHASE_0_FEASIBILITY_REPORT.md` | Feasibility findings | Phase 0 Day 5 |
| `/home/tt-admin/tt-metal/docs/architecture/adr/ADR-001-per-timestep-api.md` | Per-timestep ADR | Week 4 |
| `/home/tt-admin/tt-metal/docs/architecture/adr/ADR-002-scheduler-sync.md` | Scheduler sync ADR | Week 4 |
| `/home/tt-admin/tt-metal/docs/architecture/adr/ADR-003-controlnet-integration.md` | ControlNet ADR | Week 4 |
| `/home/tt-admin/tt-metal/docs/architecture/bridge_extension.md` | Architecture doc | Week 5 |
| `/home/tt-admin/tt-metal/docs/api/bridge_extension_api.md` | API reference | Week 5 |
| `/home/tt-admin/tt-metal/docs/guides/controlnet_guide.md` | User guide | Week 5 |
| `/home/tt-admin/tt-metal/docs/architecture/native_integration_handoff.md` | Handoff doc | Week 5 |
| `/home/tt-admin/tt-metal/docs/roadmap/PHASE_1_5_IP_ADAPTER.md` | Next phase plan | Week 5 |

---

## Config Files

| File | Purpose | Key Settings |
|------|---------|--------------|
| `/home/tt-admin/tt-metal/comfyui_bridge/config.py` | Bridge configuration | Timeout, ports, paths |
| `/home/tt-admin/tt-metal/comfyui_bridge/model_config.py` | Model definitions | Channel counts, dimensions |

---

## Key Line References

### handlers.py

| Lines | Content | Relevance |
|-------|---------|-----------|
| ~32-93 | `_detect_and_convert_tt_to_standard_format` | Format conversion helper |
| ~500-700 | `handle_denoise_only` | Existing full-loop implementation |
| ~587 | `if C != 4:` | Hardcoded channel check (to fix) |

### nodes.py

| Lines | Content | Relevance |
|-------|---------|-----------|
| TBD | `TT_KSampler` | New per-step sampler node |
| TBD | `TT_ApplyControlNet` | New ControlNet wrapper |

---

## Search Terms Index

| To Find | Search For |
|---------|------------|
| Per-step implementation | `handle_denoise_step_single`, `denoise_step` |
| Session management | `SessionManager`, `DenoiseSession`, `session_id` |
| Model configuration | `MODEL_CONFIGS`, `model_type`, `latent_channels` |
| Format conversion | `_detect_and_convert`, `standard_format`, `tt_to_standard` |
| ControlNet | `control_hint`, `ControlNet`, `TT_ApplyControlNet` |
| Scheduler sync | `timestep`, `sigma`, `scheduler` |
| Error handling | `handle_error`, `_error_response`, `recoverable` |
| Performance | `benchmark`, `latency`, `overhead` |
| SSIM validation | `ssim`, `structural_similarity`, `compare_per_step` |

---

## Quick Navigation

### By Phase

**Phase 0:**
- Checklist: `PHASE_0_EXECUTION_CHECKLIST.md`
- Validation Guide: `PHASE_0_VALIDATION_GUIDE.md`
- Go/No-Go: `PHASE_0_GO_NO_GO_FRAMEWORK.md`

**Week 1:**
- Tasks: `WEEK_1_DETAILED_TASKS.md`
- Focus: Per-step API, model config

**Week 2:**
- Tasks: `WEEK_2_DETAILED_TASKS.md`
- Focus: Session management, error handling

**Week 3:**
- Tasks: `WEEK_3_DETAILED_TASKS.md`
- Focus: ControlNet integration

**Week 4:**
- Tasks: `WEEK_4_DETAILED_TASKS.md`
- Focus: Testing, ADRs

**Week 5:**
- Tasks: `WEEK_5_DETAILED_TASKS.md`
- Focus: Validation, documentation, release

### By Role

**Lead Engineer:**
- `PHASE_1_BRIDGE_EXTENSION_PROMPT.md` - Full spec
- `WEEK_*_DETAILED_TASKS.md` - Implementation tasks
- `CODE_REVIEW_CHECKLIST.md` - Review guidance

**Support Engineer:**
- `TEST_EXECUTION_MATRIX.md` - Test schedule
- `SUCCESS_CRITERIA_FRAMEWORK.md` - Validation methods
- `PERFORMANCE_BUDGET_TRACKER.md` - Performance tracking

**Project Manager:**
- `PROGRESS_DASHBOARD_TEMPLATE.md` - Status tracking
- `WEEKLY_STATUS_TEMPLATE.md` - Reporting
- `RISK_MITIGATION_PLAYBOOK.md` - Risk management

**New Team Member:**
- `TEAM_ONBOARDING.md` - Start here
- `PHASE_1_REFERENCE_INDEX.md` - Find anything

---

## External Resources

| Resource | URL/Location | Purpose |
|----------|--------------|---------|
| ComfyUI Docs | https://docs.comfy.org/ | ComfyUI reference |
| TT-Metal Docs | Internal wiki | Hardware reference |
| ControlNet Repo | https://github.com/lllyasviel/ControlNet | ControlNet reference |

---

**Document Version:** 1.0
**Created:** December 16, 2025
