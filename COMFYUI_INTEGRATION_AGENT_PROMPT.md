# ComfyUI-Tenstorrent Integration: Strategic Analysis & Path Forward

## Executive Context

This prompt coordinates multiple specialized agents to analyze the previous ComfyUI-Tenstorrent integration attempt, understand the root causes of failure, and recommend a strategic path forward for a standalone server implementation.

## Background: The Standalone Media Server Success

**Current State** (from CURSOR_TOME.md):
- We have a **fully functional standalone SDXL media server** (`tt-media-server`)
- Built on tt-metal's proven implementations: `TTSDXLGenerateRunnerTrace`
- Architecture: Bridge-owned denoising loop with all operations in bfloat16
- Performance: ~95s total generation including model load
- Quality: High-quality image generation with SSIM ~0.65-0.69
- Key Success Factor: Complete loop control eliminates precision boundaries

**Technical Architecture**:
```
User Request → tt-media-server API → TTSDXLGenerateRunnerTrace
                                      ├─ Text Encoder (bfloat16)
                                      ├─ UNet Denoising Loop (bfloat16)
                                      │   └─ 20 steps, internal scheduler
                                      └─ VAE Decoder (bfloat16)
                                      ↓
                                   Final Image (float32)
```

## Background: The ComfyUI Integration Attempt

**What Was Attempted** (Option 3: Hybrid Bridge Approach):

### Architecture Created
1. **Backend Client** (`comfy/backends/tenstorrent_backend.py`):
   - Unix domain socket communication to bridge server
   - Shared memory tensor transfer (zero-copy)
   - <5ms latency overhead

2. **Custom Nodes** (`custom_nodes/tenstorrent_nodes/`):
   - `TT_CheckpointLoader`: Load models on TT hardware
   - `TT_FullDenoise`: Bridge-owned complete denoising loop
   - `TT_ModelInfo`: Model inspection utility
   - `TT_UnloadModel`: Explicit model cleanup

3. **Custom Samplers** (`tt_samplers.py`):
   - `sample_euler_tt`: Direct epsilon integration
   - `sample_euler_ancestral_tt`: Stochastic variant
   - Key Innovation: Avoid epsilon→denoised→epsilon roundtrip

4. **Wrappers** (`wrappers.py`):
   - `TTModelWrapper`: Routes UNet calls to bridge
   - `TTCLIPWrapper`: Routes text encoding to bridge
   - `TTVAEWrapper`: Routes VAE operations to bridge

### What Worked ✅

1. **Bridge-Owned Loop (TT_FullDenoise)**:
   - Complete success, 100% working
   - Matches tt-media-server quality
   - All denoising stays in bfloat16 until final output
   - Single request/response pattern

2. **Infrastructure**:
   - Unix socket communication: stable, low latency
   - Shared memory tensor transfer: zero-copy, efficient
   - Model loading: SDXL, SD3.5, SD1.4 all working

### What Failed ❌

**Per-Step Loop Integration (Standard KSampler)**:

**Problem 1: Loop Control Ownership Conflict**
```
tt-media-server (Working):
  Bridge owns loop → epsilon stays internal → no precision boundaries

ComfyUI Integration (Failed):
  ComfyUI owns loop → bridge called per-step → precision boundaries every step
```

**Problem 2: Numerical Precision Cascade**
- TT hardware: bfloat16 precision
- ComfyUI: float32 precision
- Conversion: bfloat16 → float32 → bfloat16 (per step!)
- At small sigma (σ < 0.5): `d = (x - denoised) / sigma` amplifies errors **33x**
- Result: Catastrophic failure at steps 16-20

**Problem 3: Scheduler State Conflicts**
- TT scheduler initialized for internal loop control
- ComfyUI scheduler expects external control
- Result: IndexError, state desynchronization

**Problem 4: The Trade-off Accepted**
The bridge-owned loop worked perfectly but sacrificed:
- ❌ ControlNet integration
- ❌ IP-Adapter support
- ❌ Custom sampler experimentation
- ❌ Alternative scheduling strategies
- ❌ ComfyUI's rich sampling ecosystem

This made it a "TT-only" execution path, defeating the purpose of ComfyUI integration.

## The Strategic Decision

**Which architectural approach best serves our goals for ComfyUI-Tenstorrent integration?**

We need to evaluate three distinct approaches across multiple dimensions:

### Option 2: Deep Native Integration
**Approach**: Implement complete TT backend directly in ComfyUI core
- Modify `model_management.py`: add `CPUState.TENSTORRENT`
- Reimplement model architectures (SDXL, SD3.5, Flux) directly in ComfyUI
- No bridge server, direct ttnn integration in ComfyUI codebase
- Native participation in ComfyUI's execution graph

### Option 3: Hybrid Bridge (Revised)
**Approach**: Improve the bridge pattern based on lessons learned
- Keep Unix socket bridge for proven model operations
- Enhance custom nodes (better ControlNet hooks, IP-Adapter support)
- Optimize the successful bridge-owned loop pattern
- Build richer TT-specific node ecosystem

### Option 2.5: Selective Native Integration
**Approach**: Strategic hybrid combining strengths of both
- Bridge for core model operations (proven, stable)
- Native hooks in ComfyUI for ecosystem features
- TT-aware sampler protocol integrated into ComfyUI core
- Incremental deepening of integration over time

### Multi-Dimensional Evaluation Criteria

**IMPORTANT**: Solving the core technical problems (precision boundaries, scheduler conflicts) is **one consideration among many**. We need comprehensive evaluation across:

#### 1. Technical Capability
- Can it deliver the features we need? (ControlNet, IP-Adapter, custom samplers)
- Does it solve, workaround, or accept the core technical challenges?
- What's the performance profile? (latency, throughput, memory)

#### 2. Development Investment
- Time to initial working prototype
- Time to feature parity with standalone server
- Time to full ecosystem integration
- Team size and skill requirements

#### 3. Maintenance & Evolution
- Ongoing maintenance burden
- Coupling to ComfyUI core updates
- Coupling to tt-metal updates
- Ability to evolve independently

#### 4. Risk Profile
- Technical risk (unproven approaches)
- Schedule risk (underestimated complexity)
- Quality risk (divergence from proven implementations)
- Organizational risk (team availability, expertise)

#### 5. Code Reuse & Quality
- Can we leverage `TTSDXLGenerateRunnerTrace` and proven code?
- Or do we reimplement, accepting validation burden?
- What's the test coverage strategy?

#### 6. Ecosystem Compatibility
- How naturally does it fit ComfyUI's architecture?
- Can standard ComfyUI workflows work unchanged?
- What compromises do users need to accept?

#### 7. Strategic Flexibility
- Can we build incrementally or require big-bang?
- Can we pivot if assumptions prove wrong?
- Does it position us well for future models (Flux, SD4, etc.)?

#### 8. User Experience
- Workflow ergonomics (node complexity, debugging)
- Performance (is 1-5ms bridge overhead acceptable?)
- Feature availability (which ecosystem features work?)

**Critical Insight**: An approach that accepts certain technical limitations but excels in development time, maintenance, code reuse, and risk management may be superior to one that "solves" technical problems at high cost.

## Task Assignments for Specialized Agents

### 1. Problem-Investigator Agent

**Task**: Root Cause Analysis & Constraint Mapping

Investigate the following:

a) **Precision Boundary Analysis**:
   - Where exactly do bfloat16↔float32 conversions happen in the failed integration?
   - Are these conversions required by ComfyUI's architecture or the bridge pattern?
   - Review ComfyUI core files: `comfy/model_management.py`, `comfy/samplers.py`
   - **For each option**: Can it eliminate, workaround, or must accept these boundaries?
   - What are the practical implications of each approach?

b) **Scheduler State Conflict Deep Dive**:
   - Examine ComfyUI's sampler calling pattern (per-step vs batch)
   - Compare with tt-metal's scheduler design (self-contained loop)
   - Identify if this is a fundamental architectural incompatibility
   - **For each option**: How does it handle this constraint?
   - Could we create a "TT-aware sampler protocol" in ComfyUI? What would that require?

c) **Success Pattern Analysis**:
   - Why did `TT_FullDenoise` work perfectly?
   - What specific architectural decisions made bridge-owned loops succeed?
   - Can this pattern be generalized to support ControlNet/IP-Adapter?
   - What's the cost/benefit of building on this proven pattern vs starting fresh?

d) **Failure Mode Catalog & Resolution Strategies**:
   - Document all failure modes from the previous attempt
   - Classify: Bridge-specific vs Fundamental ComfyUI constraint
   - **For each failure**: How would Option 2, Option 3 (revised), and Option 2.5 handle it?
   - Rate: Solves completely / Workable workaround / Must accept limitation

e) **Constraint vs Flexibility Analysis**:
   - Which technical constraints are truly immovable?
   - Which are design choices we can influence or work around?
   - What creative solutions might we be missing?

**Deliverable**: A comprehensive root cause analysis document providing:
- Technical constraint map (immovable vs negotiable)
- Failure resolution matrix (how each option handles each challenge)
- Creative solution catalog (workarounds, compromises, innovations)
- Practical implications assessment (what do users experience in each scenario?)

---

### 2. Knowledge-Curator Agent

**Task**: ComfyUI Architecture Deep Research

Research and document:

a) **ComfyUI Sampling Architecture**:
   - How does `comfy/samplers.py` work?
   - What is the contract for custom samplers?
   - Can samplers own their loops while remaining compatible?
   - Review: `k_diffusion` integration, CFGGuider design

b) **ComfyUI Extension Patterns**:
   - How do ControlNet nodes work? (`custom_nodes/*/controlnet`)
   - How do IP-Adapter nodes integrate? (if available)
   - What hooks exist for custom processing paths?
   - Can we inject TT-specific paths without breaking compatibility?

c) **Precision Requirements in ComfyUI**:
   - Where is float32 mandated vs flexible?
   - Review tensor operations in `comfy/model_base.py`
   - Can we create bfloat16 paths in ComfyUI core?

d) **Backend Pattern Analysis**:
   - Does ComfyUI support multiple backends? (CPU, CUDA, DirectML)
   - How are backend-specific optimizations implemented?
   - Can we reference similar integrations as templates?

e) **Custom Node Ecosystem**:
   - How do popular custom node packs structure complex integrations?
   - Examples: ComfyUI-Manager, AnimateDiff, Reactor
   - What lessons can we apply?

**Deliverable**: A knowledge base document covering:
- ComfyUI's sampler architecture and extension points
- Precision handling and backend integration patterns
- Best practices from successful custom node integrations
- Technical constraints and opportunities

---

### 3. Integration-Orchestrator Agent

**Task**: Multi-Dimensional Strategic Analysis & Synthesis

Synthesize findings from problem-investigator and knowledge-curator to provide comprehensive evaluation:

a) **Feature Feasibility Matrix** (All Three Options):

For each desired feature, evaluate ALL options:

| Feature | Option 2 (Native) | Option 3 (Bridge) | Option 2.5 (Hybrid) |
|---------|-------------------|-------------------|---------------------|
| Custom Checkpoint Loading | ? | ? | ? |
| Custom KSampler | ? | ? | ? |
| ControlNet Support | ? | ? | ? |
| IP-Adapter Support | ? | ? | ? |
| Alternative Schedulers | ? | ? | ? |
| Custom Samplers | ? | ? | ? |
| Flux/SD4 Future Models | ? | ? | ? |

For each cell: **Feasibility** (Easy/Medium/Hard/Blocked), **Approach** (1 sentence), **Quality** (matches tt-media-server?)

b) **Multi-Dimensional Comparison Matrix**:

| Criterion | Weight | Option 2 | Option 3 | Option 2.5 | Notes |
|-----------|--------|----------|----------|------------|-------|
| **Technical Capability** |
| - Solves core problems | ? | 1-10 | 1-10 | 1-10 | Is this decisive? |
| - Performance (latency) | ? | 1-10 | 1-10 | 1-10 | Is 1-5ms acceptable? |
| - Feature completeness | ? | 1-10 | 1-10 | 1-10 | What can users do? |
| **Development & Risk** |
| - Time to MVP | ? | 1-10 | 1-10 | 1-10 | Weeks or months? |
| - Time to parity | ? | 1-10 | 1-10 | 1-10 | Full feature set? |
| - Technical risk | ? | 1-10 | 1-10 | 1-10 | Unproven approaches? |
| - Schedule risk | ? | 1-10 | 1-10 | 1-10 | Hidden complexity? |
| **Maintenance & Evolution** |
| - Code reuse | ? | 1-10 | 1-10 | 1-10 | Proven implementations? |
| - Maintenance burden | ? | 1-10 | 1-10 | 1-10 | Ongoing cost? |
| - ComfyUI coupling | ? | 1-10 | 1-10 | 1-10 | Update fragility? |
| - tt-metal coupling | ? | 1-10 | 1-10 | 1-10 | Update fragility? |
| **Strategic Flexibility** |
| - Incremental build | ? | 1-10 | 1-10 | 1-10 | Can we ship early? |
| - Pivot capability | ? | 1-10 | 1-10 | 1-10 | If assumptions wrong? |
| - Future extensibility | ? | 1-10 | 1-10 | 1-10 | New models, features? |
| **TOTAL SCORE** | | ? | ? | ? | |

c) **Option 2.5 Design Exploration**:

Design THREE specific Option 2.5 variants, each with different trade-offs:

**Variant A: "Bridge-First with Native Hooks"**
- Core: Keep full bridge pattern
- Addition: Native ControlNet/IP-Adapter hooks in ComfyUI
- Tradeoff: Accepts 1-5ms latency, minimizes development risk
- Best for: Fast delivery, proven stability

**Variant B: "Native Core with Bridge Fallback"**
- Core: Native TT backend in ComfyUI
- Addition: Bridge for complex models until native catches up
- Tradeoff: Higher development cost, smoother migration path
- Best for: Long-term native integration, phased approach

**Variant C: "Selective Native Integration"**
- Core: Bridge for models, native for schedulers/samplers
- Addition: TT-aware sampler protocol in ComfyUI core
- Tradeoff: Splits concerns, requires ComfyUI core changes
- Best for: Balancing control and reuse

For each variant:
- Architecture diagram
- What's native vs bridged?
- Development phases
- Risk assessment

d) **Sensitivity Analysis**:
- If development time is the top constraint → Recommended option?
- If ecosystem features are must-haves → Recommended option?
- If maintenance burden must be minimal → Recommended option?
- If technical risk must be minimal → Recommended option?

e) **Dependency Chain & Parallel Work**:
- Critical path identification
- What can be built in parallel?
- Early validation checkpoints
- Incremental delivery milestones

**Deliverable**: A comprehensive strategic analysis providing:
- Multi-dimensional comparison with scored matrix
- Three concrete Option 2.5 variants with designs
- Sensitivity analysis for different priorities
- Phased implementation plan for each option
- Clear trade-off articulation (what you get vs what you give up)
- **Recommendation framework** (not a single recommendation, but "If X matters most, choose Y")

---

### 4. Communications-Translator Agent

**Task**: Strategic Decision Framework & Multi-Audience Communication

Translate technical findings into clear, actionable strategic guidance:

a) **Executive Summary** (1-2 pages):
- What failed in the previous attempt and why? (2-3 paragraphs)
- What did we learn? (key insights, not just problems)
- What are our real options? (Option 2, 3, 2.5 variants)
- What's the decision framework? (how to choose based on priorities)

b) **Decision Framework Document**:

**1. The Options Landscape**:
Present each option clearly with:
- **What it is**: 2-sentence description
- **Best for**: What priorities does it optimize?
- **Accepts**: What constraints/limitations does it live with?
- **Delivers**: What capabilities does it enable?
- **Timeline**: Rough milestones (MVP, parity, full feature)
- **Risk profile**: What could go wrong?

**2. Decision Matrix for Leadership**:

```
If your priority is...          → Choose...        → Because...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fastest time to market          → Option 3/2.5-A   → Proven pattern, low risk
Full ComfyUI ecosystem support  → Option 2.5-C     → Native sampler protocol
Lowest maintenance burden       → Option 3         → Isolated bridge, clear boundary
Maximum technical elegance      → Option 2         → Native integration, no compromise
Balanced risk/reward            → Option 2.5-A/C   → Hybrid strengths
Future-proofing                 → Option 2.5-B     → Migration path to native
```

**3. Trade-off Clarity**:

For EACH option, be explicit about:
- ✅ **What you GET**: Specific capabilities, benefits
- ❌ **What you GIVE UP**: Specific limitations, costs
- ⚖️ **What you ACCEPT**: Constraints you work within
- 🎯 **What you OPTIMIZE FOR**: Primary goals

**4. Implementation Roadmap** (for each viable option):

Provide phase breakdown:
- **Phase 1**: What / Why / Success Criteria / Duration
- **Phase 2**: What / Why / Success Criteria / Duration
- **Phase 3**: What / Why / Success Criteria / Duration
- **Phase 4**: What / Why / Success Criteria / Duration

**Validation strategy**:
- How do we know quality matches tt-media-server?
- What are early warning signs of wrong path?
- What are the pivot/kill decision points?

**5. Risk & Mitigation**:
Not just "here are risks," but:
- **Risk**: Specific concern
- **Impact**: What happens if it materializes?
- **Likelihood**: Based on evidence
- **Mitigation**: Concrete actions to reduce/manage
- **Fallback**: Plan B if mitigation fails

c) **Multi-Audience Communication**:

**For Technical Team**:
- Architecture diagrams for each option
- API contracts and integration points
- Code structure and organization
- Testing and validation strategy
- Clear technical trade-offs with rationale

**For Product/Project Management**:
- Feature delivery timeline by option
- User-visible capabilities and limitations
- Ecosystem compatibility matrix
- Resource requirements (team, time, infrastructure)
- Dependencies and blockers

**For Executive/Strategic Decision Makers**:
- Business case for each option (value, cost, risk)
- Strategic positioning (market, ecosystem, innovation)
- Resource commitment and opportunity cost
- Go/no-go decision framework
- Success metrics and accountability

d) **Recommendation Approach**:

**IMPORTANT**: Don't provide a single "we recommend X" conclusion. Instead:

1. **Present the trade-off space clearly**
2. **Show how different priorities → different choices**
3. **Provide decision framework** for stakeholders to apply their priorities
4. **Highlight default recommendation** based on balanced priorities, BUT
5. **Clearly state**: "This recommendation assumes [X, Y, Z priorities]. If your priorities differ, the choice may differ."

**Example closing**:
> "If we optimize for **speed and risk mitigation**, Option 2.5-A (Bridge-First with Native Hooks) is the clear choice. However, if **long-term native integration** is the paramount goal and we accept higher upfront cost, Option 2.5-B provides a smoother migration path. The decision ultimately depends on whether we prioritize shipping quickly vs. long-term architectural purity."

**Deliverable**: A polished strategic document suite that:
- **Empowers decision-making** with clear frameworks
- **Presents options honestly** with real trade-offs
- **Adapts to audience** (technical, product, executive)
- **Enables confident choices** based on priorities
- **Provides actionable guidance** for implementation
- **Manages expectations** realistically about constraints and timelines

---

## Required Context & Resources

### Available Repositories
- `/home/tt-admin/tt-metal`: Standalone SDXL server (working, proven)
- `/home/tt-admin/ComfyUI-tt`: Previous integration attempt (mixed success)
- `/home/tt-admin/ComfyUI-tt_standalone`: Clean ComfyUI base for new work

### Key Files to Review

**Working Standalone Server**:
- `tt-metal/tt_sdxl_pipeline.py`: Core pipeline
- `tt-metal/sdxl_runner.py`: Runner implementation
- `tt-metal/models/experimental/stable_diffusion_xl_base/`: Model implementations

**Previous ComfyUI Attempt**:
- `ComfyUI-tt/comfy/backends/tenstorrent_backend.py`: Backend client
- `ComfyUI-tt/custom_nodes/tenstorrent_nodes/nodes.py`: Custom nodes
- `ComfyUI-tt/custom_nodes/tenstorrent_nodes/tt_samplers.py`: Custom samplers
- `ComfyUI-tt/custom_nodes/tenstorrent_nodes/wrappers.py`: Model wrappers

**ComfyUI Core** (clean base):
- `ComfyUI-tt_standalone/comfy/samplers.py`: Sampler architecture
- `ComfyUI-tt_standalone/comfy/model_management.py`: Backend integration
- `ComfyUI-tt_standalone/nodes.py`: Standard nodes

### Documentation to Reference
- `CURSOR_TOME.md`: Standalone server debugging journey
- `PICKUP_COMFYUI.md`: Previous integration summary
- `ComfyUI-tt/custom_nodes/tenstorrent_nodes/TT_SAMPLERS_ARCHITECTURE.md`: Detailed sampler design

### Key Questions Each Agent Must Address

**All Agents - Multi-Dimensional Thinking**:
1. For each option (2, 3, 2.5 variants): What are the **complete trade-offs**, not just technical capability?
2. How do we balance: Speed vs Quality? Risk vs Reward? Short-term vs Long-term?
3. What assumptions are we making? How do we validate them early?
4. What does "success" really mean? (Not just "works" but "delivers value")

**Problem-Investigator - Constraint & Possibility Mapping**:
5. Are the failures bridge-specific, ComfyUI constraints, or TT-specific?
6. Which constraints are immovable vs negotiable?
7. What creative workarounds might we be missing?
8. For per-step calling: Is it truly impossible, or just harder than alternatives?
9. What's the practical impact of technical limitations on users?

**Knowledge-Curator - Ecosystem & Pattern Research**:
10. What extension points exist in ComfyUI for deep integration?
11. How have other hardware backends (DirectML, ONNX, TensorRT) approached this?
12. What patterns do successful complex custom nodes use?
13. What's the cost/benefit of ComfyUI core modifications vs external integration?
14. Can we learn from other "foreign hardware" integrations in ML frameworks?

**Integration-Orchestrator - Strategic Synthesis**:
15. What's the critical path for each option to deliver user value?
16. Can we build incrementally or require big-bang integration?
17. How sensitive is the recommendation to different priority weightings?
18. What's the "minimum viable integration" for each approach?
19. Where are the decision points for pivoting between options?
20. What does Option 2.5 really mean? (Design concrete variants, not abstractions)

**Communications-Translator - Decision Enablement**:
21. How do we present options without bias toward "solving problems" as the only goal?
22. What's the decision framework for stakeholders with different priorities?
23. How do we communicate uncertainty and trade-offs honestly?
24. What does each option mean for users? (Not just developers)
25. What's the "line in the sand" - which features are must-haves vs nice-to-haves?

**Meta-Question for All Agents**:
> "If development speed is critical and we can only ship in 2 months, does that change our recommendation compared to having 6 months? How?"

This question forces thinking beyond pure technical merit.

## Success Criteria for This Analysis

This prompt succeeds when we have:

1. ✅ **Multi-Dimensional Understanding**
   - Clear technical constraint map (what's immovable, what's negotiable)
   - How each option handles each constraint (solve, workaround, accept)
   - Practical implications for users and developers

2. ✅ **Comprehensive Option Evaluation**
   - ALL options evaluated across ALL dimensions (not just technical capability)
   - Feature feasibility matrix for Options 2, 3, and 2.5 variants
   - Honest assessment of trade-offs (what you get vs what you give up)

3. ✅ **Decision Framework (Not Single Recommendation)**
   - Clear framework: "If X priority, choose Y option, because Z"
   - Sensitivity analysis showing how priorities change recommendations
   - Concrete Option 2.5 variants (not vague "hybrid" concept)

4. ✅ **Implementation Guidance for Each Viable Option**
   - Phased plan with clear milestones
   - Risk mitigation strategies
   - Early validation checkpoints
   - Pivot/kill decision points

5. ✅ **Feature Feasibility Matrix (All Options)**
   ```
   | Feature | Option 2 | Option 3 | Option 2.5-A | Option 2.5-B | Option 2.5-C |
   |---------|----------|----------|--------------|--------------|--------------|
   | Custom Checkpoint | ?/?/? | ?/?/? | ?/?/? | ?/?/? | ?/?/? |
   | Custom KSampler | ?/?/? | ?/?/? | ?/?/? | ?/?/? | ?/?/? |
   | ControlNet | ?/?/? | ?/?/? | ?/?/? | ?/?/? | ?/?/? |
   | IP-Adapter | ?/?/? | ?/?/? | ?/?/? | ?/?/? | ?/?/? |
   | Custom Samplers | ?/?/? | ?/?/? | ?/?/? | ?/?/? | ?/?/? |
   | Alt Schedulers | ?/?/? | ?/?/? | ?/?/? | ?/?/? | ?/?/? |
   ```
   Format: Feasibility/Quality/Effort (e.g., "Easy/Full/Low")

6. ✅ **Honest Communication of Uncertainty**
   - What do we know vs assume?
   - What needs early validation?
   - Where could we be wrong?

7. ✅ **Multi-Audience Deliverables**
   - Technical teams can implement from this
   - Product/PM can plan from this
   - Executives can decide from this

## Coordination Protocol

1. **Problem-Investigator** runs first (root cause analysis)
2. **Knowledge-Curator** runs in parallel or after (ComfyUI research)
3. **Integration-Orchestrator** synthesizes findings (feasibility analysis)
4. **Communications-Translator** produces final strategic document

Each agent should:
- Reference findings from previous agents
- Highlight uncertainties requiring follow-up
- Provide concrete evidence for claims
- Suggest specific code/files to review for validation

## Expected Outputs

### From Problem-Investigator
- Root cause analysis: 3-5 pages
- Failure taxonomy: Bridge-specific vs Fundamental
- Option 2 capability matrix: What can/can't it solve?

### From Knowledge-Curator
- ComfyUI architecture guide: 4-6 pages
- Extension points catalog
- Backend integration patterns
- Best practice recommendations

### From Integration-Orchestrator
- Option comparison matrix with scores
- Recommended approach with architecture
- Phased implementation plan
- Risk analysis and mitigation

### From Communications-Translator
- Executive summary: 1-2 pages
- Strategic recommendation: 3-5 pages
- Technical implementation guide: 5-8 pages
- Stakeholder communication materials

## Final Deliverable

A comprehensive strategic framework that enables us to:

1. **Understand the Trade-off Space**
   - What are our real options? (Not just 2 vs 3, but specific variants)
   - What does each option optimize for?
   - What constraints does each option accept?

2. **Make Priority-Aligned Decisions**
   - Decision framework: "If X matters most → Choose Y → Because Z"
   - Sensitivity analysis: How do different priorities change the choice?
   - Not a single recommendation, but a principled way to choose

3. **Plan Implementation Realistically**
   - Phased roadmap for each viable option
   - Resource requirements (team, time, infrastructure)
   - Early validation checkpoints
   - Pivot/kill decision points

4. **Execute with Confidence**
   - Clear implementation guidance with success criteria
   - Risk mitigation strategies
   - What does "done" mean at each phase?

5. **Adapt as We Learn**
   - What assumptions need early validation?
   - What are signs we've chosen the wrong path?
   - When and how do we pivot?

6. **Communicate Effectively**
   - Technical teams: What to build and how
   - Product/PM: What users get and when
   - Executives: Strategic value, cost, risk

---

**Agent Coordination Note**: This is a complex multi-agent task requiring synthesis across technical, strategic, and organizational dimensions.

**Each agent should clearly identify**:
- ✅ **Conclusions supported by evidence** (with citations)
- ❓ **Uncertainties requiring investigation** (with proposed validation methods)
- 🔄 **Assumptions requiring validation** (with early test strategies)
- 🚨 **Red flags or blocking issues** (with severity assessment)
- ⚖️ **Trade-offs** (explicit: getting X means accepting Y)

**The Integration-Orchestrator is responsible for**:
- Synthesizing findings into coherent evaluation
- Resolving conflicts between agents
- Designing concrete Option 2.5 variants (not leaving it abstract)
- Building the decision framework (not making a single recommendation)
- Ensuring all dimensions are considered (not just technical capability)

**Critical Success Factor**: The final output should empower decision-makers to choose confidently based on their priorities, not tell them what to choose.
