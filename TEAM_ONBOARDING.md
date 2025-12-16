# Team Onboarding Package

**Project:** Phase 1 Bridge Extension  
**Duration:** 5-6 weeks  
**Start Date:** _______________

---

## Quick Overview (1 Page)

### What Are We Building?

We are extending the ComfyUI bridge to support **per-timestep denoising**, which enables:
- **ControlNet** integration (pose-guided, edge-guided generation)
- **IP-Adapter** integration (image-prompted generation)
- **Custom samplers** (advanced sampling methods)

### Why This Approach?

1. **Bridge Extension vs Native Integration**
   - Bridge: 5-6 weeks, validates patterns
   - Native: 12-17 weeks, larger investment
   - Decision: Bridge first, then native

2. **Per-Timestep vs Full-Loop**
   - Full-loop runs internally, blocks extensions
   - Per-timestep enables external control
   - Required for ComfyUI ecosystem features

### Key Success Metrics

| Metric | Target |
|--------|--------|
| Per-step SSIM vs full-loop | >= 0.99 |
| ControlNet SSIM vs CPU | >= 0.90 |
| Human validation | 5/5 correct |
| Performance overhead | < 10% |
| Robustness | 1000 gen, 0 crashes |

### Timeline Overview

```
Phase 0 (5 days):     Validate assumptions
Week 1:               Per-timestep API foundation
Week 2:               Session management
Week 3:               ControlNet implementation
Week 4:               Testing and ADRs
Week 5:               Final validation and docs
Buffer:               3-5 days
```

---

## Reference Documents Map

### Primary Documents

| Document | Purpose | Location |
|----------|---------|----------|
| Master Prompt | Complete implementation spec | `/home/tt-admin/tt-metal/PHASE_1_BRIDGE_EXTENSION_PROMPT.md` |
| Phase 0 Checklist | Daily tasks for validation | `/home/tt-admin/tt-metal/PHASE_0_EXECUTION_CHECKLIST.md` |
| Week 1-5 Tasks | Detailed daily breakdown | `/home/tt-admin/tt-metal/WEEK_[1-5]_DETAILED_TASKS.md` |

### Supporting Documents

| Document | Purpose | Location |
|----------|---------|----------|
| Success Criteria | How to measure success | `/home/tt-admin/tt-metal/SUCCESS_CRITERIA_FRAMEWORK.md` |
| Risk Playbook | How to handle risks | `/home/tt-admin/tt-metal/RISK_MITIGATION_PLAYBOOK.md` |
| Reference Index | All file locations | `/home/tt-admin/tt-metal/PHASE_1_REFERENCE_INDEX.md` |

### Templates

| Template | When to Use | Location |
|----------|-------------|----------|
| Daily Standup | Every morning | `/home/tt-admin/tt-metal/DAILY_STANDUP_TEMPLATE.md` |
| Weekly Review | End of each week | `/home/tt-admin/tt-metal/WEEKLY_REVIEW_CHECKLIST.md` |
| Code Review | Before merging code | `/home/tt-admin/tt-metal/CODE_REVIEW_CHECKLIST.md` |

### Strategic Context

| Document | Purpose | Location |
|----------|---------|----------|
| Parity Status | Current state analysis | `/home/tt-admin/tt-metal/DEC15_PARITY_STATUS_CORRECTION.md` |
| Path Analysis | Why bridge first | `/home/tt-admin/tt-metal/STRATEGIC_PATH_ANALYSIS.md` |
| Bridge to Native | Integration pathway | `/home/tt-admin/tt-metal/BRIDGE_TO_INTEGRATION_ANALYSIS.md` |

---

## Team Member Roles

| Role | Responsibilities | Primary Contact |
|------|-----------------|-----------------|
| Lead Engineer | Implementation, technical decisions | |
| Support Engineer | Testing, documentation | |
| Technical Lead | Architecture review, escalation | |
| Project Manager | Timeline, resources, stakeholders | |

### Role-Specific Focus

**Lead Engineer:**
- Primary implementation (handlers.py, session_manager.py)
- Technical design decisions
- ADR authorship
- Code review

**Support Engineer:**
- Test implementation
- Documentation
- Performance benchmarking
- Integration testing

---

## Where to Find Information

### By Topic

| Topic | Primary Source | Secondary Source |
|-------|---------------|------------------|
| What to build | Master Prompt (Part C) | Week detailed tasks |
| How to measure | Success Criteria Framework | Master Prompt (Part E) |
| What could go wrong | Risk Playbook | Master Prompt (Part F) |
| Technical decisions | ADRs (after creation) | Master Prompt (Part D) |
| Code locations | Reference Index | Master Prompt (File References) |

### By Question Type

| Question | Where to Look |
|----------|---------------|
| "What should I work on today?" | Phase 0 Checklist or Week N tasks |
| "How do I know if this is done?" | Success Criteria Framework |
| "Something went wrong, what do I do?" | Risk Mitigation Playbook |
| "Where is the code for X?" | Reference Index |
| "Why did we decide Y?" | ADRs or Master Prompt (Part D) |

---

## How to Ask Questions

### Self-Service First

1. Check the Reference Index for file locations
2. Check the Master Prompt for specifications
3. Check ADRs for design rationale
4. Check existing code comments

### When to Ask Team

- Implementation ambiguity not covered in docs
- Risk situation requiring escalation
- Resource or timeline concerns
- Technical blockers

### How to Ask

**Slack/Teams Format:**
```
Topic: [Brief topic]
Question: [Specific question]
Context: [What you've already checked/tried]
Urgency: [Today/This Week/Eventually]
```

**Example:**
```
Topic: Scheduler sync design
Question: Should we validate sigma values against a known schedule?
Context: Checked ADR-002, mentions stateless but not validation
Urgency: Today (blocking Day 2 task)
```

---

## Escalation Paths

### Technical Issues

```
1. Check documentation
       |
       v
2. Ask team member (Slack)
       |
       v
3. Schedule sync with Lead Engineer
       |
       v
4. Escalate to Technical Lead
```

### Resource/Timeline Issues

```
1. Document impact in standup
       |
       v
2. Discuss in weekly review
       |
       v
3. Escalate to Project Manager
```

### Risk Materialization

```
1. Follow Risk Playbook
       |
       v
2. Escalate per playbook guidance
       |
       v
3. Document in decision log
```

---

## Key Contacts

| Role | Name | Contact | Best For |
|------|------|---------|----------|
| Lead Engineer | | | Technical questions |
| Technical Lead | | | Architecture decisions |
| Project Manager | | | Timeline, resources |
| DevOps | | | Environment issues |

---

## First Day Checklist

- [ ] Read this onboarding document
- [ ] Skim Master Prompt (focus on Parts A, B, C)
- [ ] Review Phase 0 or current week tasks
- [ ] Set up development environment
- [ ] Run existing test suite to verify setup
- [ ] Join project Slack/Teams channel
- [ ] Schedule intro sync with Lead Engineer

---

## Development Environment Setup

### Prerequisites

- Python 3.10+
- TT-Metal SDK installed
- ComfyUI installed at `/home/tt-admin/ComfyUI-tt_standalone/`
- Bridge code at `/home/tt-admin/tt-metal/comfyui_bridge/`

### Verification

```bash
# Verify bridge runs
cd /home/tt-admin/tt-metal
python -m comfyui_bridge.server --help

# Run existing tests
pytest comfyui_bridge/tests/ -v

# Verify ComfyUI custom nodes
ls /home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/
```

---

## Key Files to Know

### Bridge Implementation

| File | Purpose | Modify? |
|------|---------|---------|
| `comfyui_bridge/handlers.py` | Main operation handlers | Yes |
| `comfyui_bridge/server.py` | Server entry point | Maybe |
| `comfyui_bridge/model_config.py` | Model configurations | Create |
| `comfyui_bridge/session_manager.py` | Session management | Create |

### ComfyUI Nodes

| File | Purpose | Modify? |
|------|---------|---------|
| `custom_nodes/tenstorrent_nodes/nodes.py` | ComfyUI node definitions | Yes |
| `custom_nodes/tenstorrent_nodes/__init__.py` | Node registration | Maybe |

### Tests

| File | Purpose | Modify? |
|------|---------|---------|
| `comfyui_bridge/tests/test_per_step.py` | Per-step API tests | Create |
| `comfyui_bridge/tests/test_controlnet.py` | ControlNet tests | Create |
| `comfyui_bridge/tests/test_session.py` | Session tests | Create |

---

## Common Commands

```bash
# Run tests
pytest comfyui_bridge/tests/ -v

# Run specific test
pytest comfyui_bridge/tests/test_per_step.py -v

# Run performance benchmark
python comfyui_bridge/tests/benchmark_per_step.py

# Start bridge server
python -m comfyui_bridge.server

# Check code style
python -m flake8 comfyui_bridge/
```

---

## Quick Reference

### Success Criteria Summary

- SSIM >= 0.99 for per-step vs full-loop
- SSIM >= 0.90 for ControlNet vs CPU
- 5/5 human raters confirm correct
- < 10% latency overhead
- 1000 generations, 0 crashes

### Key Design Decisions

1. **Stateless Bridge:** ComfyUI owns scheduler, bridge receives timestep/sigma
2. **Model-Agnostic:** Config-based channel lookup, not hardcoded
3. **CPU-Side ControlNet:** ControlNet runs on ComfyUI side, hint passed to bridge
4. **Session Management:** Dict-based with 30-min timeout

### Important Deadlines

| Milestone | Date | Deliverable |
|-----------|------|-------------|
| Phase 0 Complete | Day 5 | Go/No-Go decision |
| Week 2 Complete | | Session management working |
| Week 3 Complete | | ControlNet working |
| Week 5 Complete | | Release ready |

---

**Welcome to the team! Questions? Ask in the project channel.**

---

**Document Version:** 1.0  
**Created:** December 16, 2025
