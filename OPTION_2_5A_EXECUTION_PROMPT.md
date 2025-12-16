# Option 2.5A Implementation Execution Prompt

## Objective
Execute the Option 2.5A implementation plan using available Claude Code sub-agents optimally to deliver ComfyUI-Tenstorrent integration in phases.

---

## Available Sub-Agents and Optimal Usage

### 1. **Explore Agent** - Codebase Investigation
**Use for:**
- Understanding existing architecture before making changes
- Finding all files related to a specific component
- Mapping dependencies between modules
- Answering "where is X handled?" questions

### 2. **Problem-Investigator Agent** - Root Cause Analysis
**Use for:**
- Phase 0 critical validation (timestep bug, CFG batching)
- Debugging issues that arise during implementation
- Investigating performance bottlenecks
- Analyzing test failures

### 3. **Code-Writer Agent** - Implementation
**Use for:**
- Writing new features and components
- Refactoring existing code
- Implementing fixes identified by problem-investigator
- Adding new nodes, handlers, wrappers

### 4. **Critical-Reviewer Agent** - Quality Assurance
**Use for:**
- Reviewing implemented code before committing
- Checking for security vulnerabilities
- Validating test coverage
- Ensuring code follows best practices

### 5. **Local-File-Searcher Agent** - Targeted Code Search
**Use for:**
- Finding specific function implementations
- Locating where a variable/class is used
- Gathering context for code modifications
- Quick reference lookups

---

## Phase 0: Critical Validation (Week 1)

### Task 0.1: Timestep Format Bug Investigation

**Agent Sequence:**

```markdown
1. **Problem-Investigator** (Primary)
   Prompt: "Investigate the timestep format bug in ComfyUI-TT integration.

   **Problem Statement:**
   ComfyUI passes continuous sigma values (e.g., 14.6146) but TT scheduler
   expects discrete timestep indices (e.g., 999).

   **Files to investigate:**
   - /home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/wrappers.py (lines 340-360)
   - /home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/tt_euler_discrete_scheduler.py

   **Tasks:**
   1. Add comprehensive logging to capture actual timestep values
   2. Run single inference with 20 steps
   3. Capture log output showing timestep values at each step
   4. Determine if there's a format mismatch
   5. Assess severity (blocker/fixable/non-issue)
   6. Propose fix if needed

   **Success Criteria:**
   - Clear evidence of mismatch or confirmation it's working
   - Fix strategy with estimated effort
   - SSIM >= 0.90 after fix"

2. **Code-Writer** (If fix needed)
   Prompt: "Implement the timestep conversion fix identified in the investigation.

   [Paste problem-investigator findings]

   Requirements:
   - Implement sigma-to-timestep conversion in wrappers.py
   - Add unit tests for conversion function
   - Ensure backward compatibility
   - Add logging for verification"

3. **Critical-Reviewer** (Validation)
   Prompt: "Review the timestep format fix implementation.

   Check for:
   - Correctness of conversion logic
   - Edge cases (first step, last step, non-standard step counts)
   - Performance impact of conversion
   - Test coverage adequacy
   - Potential precision issues"
```

### Task 0.2: CFG Batching Verification

**Agent Sequence:**

```markdown
1. **Problem-Investigator** (Primary)
   Prompt: "Verify CFG batching compatibility between ComfyUI and TT UNet.

   **Problem Statement:**
   ComfyUI batches conditional/unconditional as [uncond, cond] for CFG.
   Need to verify TT UNet processes this correctly.

   **Files to investigate:**
   - /home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/wrappers.py (lines 310-335)
   - /home/tt-admin/ComfyUI-tt_standalone/comfy/samplers.py (lines 298-326)
   - TT UNet forward pass implementation

   **Tasks:**
   1. Add logging to capture batch shapes and statistics
   2. Verify batch order [uncond, cond]
   3. Compare output with separate uncond/cond calls
   4. Check if TT-side CFG conflicts with ComfyUI CFG
   5. Measure quality difference (SSIM/visual inspection)

   **Success Criteria:**
   - Confirmation of correct batch processing
   - No quality degradation vs separate passes
   - Fix strategy if issues found"

2. **Code-Writer** (If fix needed)
   Prompt: "Implement CFG batching fix based on investigation findings.

   [Paste problem-investigator findings]

   Options to implement:
   - Batch reordering if order is wrong
   - Separate calls if batching incompatible
   - TT-side CFG disable if conflict exists"
```

### Task 0.3: Go/No-Go Decision Document

**Agent Sequence:**

```markdown
1. **Integration-Orchestrator** (Synthesize results)
   Prompt: "Create Phase 0 Go/No-Go decision document based on validation results.

   **Inputs:**
   - Timestep bug investigation results
   - CFG batching verification results
   - SSIM scores from validation tests

   **Required Analysis:**
   1. Summarize findings from both investigations
   2. Assess overall risk level (low/medium/high)
   3. Recommend path forward:
      - GO: Proceed to Phase 1 with per-step integration
      - CONDITIONAL GO: Proceed with modifications
      - NO-GO: Pivot to bridge-only architecture
   4. Document required changes and timeline impact

   **Deliverable Format:**
   - Executive summary (1 paragraph)
   - Findings table (timestep, CFG status)
   - Risk assessment
   - Recommendation with rationale
   - Updated timeline if needed"
```

---

## Phase 1: Bridge Stabilization (Weeks 2-5)

### Task 1.1: Apply Phase 0 Fixes and SSIM Regression Investigation

**Agent Sequence:**

```markdown
1. **Explore** (Context gathering)
   Prompt: "Explore the SDXL pipeline to understand recent changes affecting SSIM.

   Context from CURSOR_TOME.md: SSIM regressed from ~0.69 to ~0.65 after merge.

   Find:
   - Recent changes to sdxl_runner.py batch processing
   - L1_SMALL size changes in sdxl_config.py
   - guidance_rescale parameter changes
   - Any tensor precision changes

   Map the changes and their potential impact on image quality."

2. **Problem-Investigator** (Root cause analysis)
   Prompt: "Investigate SSIM regression in SDXL pipeline.

   **Context:** [Paste Explore findings]

   **Tasks:**
   1. Generate baseline images with known seeds (pre-merge commit)
   2. Generate comparison images with current code
   3. Calculate SSIM scores
   4. Run PCC tests on UNet intermediate outputs
   5. Identify which change caused regression
   6. Propose fix or document as acceptable if minor

   **Success Criteria:**
   - Root cause identified
   - SSIM >= 0.95 target or explanation why current is acceptable"

3. **Code-Writer** (Fix implementation)
   Prompt: "Implement SSIM regression fix based on investigation.

   [Paste problem-investigator findings]

   Implement the recommended fix and add regression test to prevent future issues."

4. **Critical-Reviewer** (Validation)
   Prompt: "Review SSIM fix implementation.

   Verify:
   - Fix addresses root cause
   - No performance degradation
   - Regression test added
   - SSIM targets met"
```

### Task 1.2: Production Error Handling

**Agent Sequence:**

```markdown
1. **Local-File-Searcher** (Find error handling patterns)
   Prompt: "Find all error handling in the TenstorrentBackend and bridge code.

   Search for:
   - try/except blocks
   - timeout handling
   - connection error handling
   - resource cleanup in finally blocks

   Files to search:
   - /home/tt-admin/ComfyUI-tt/comfy/backends/tenstorrent_backend.py
   - /home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/nodes.py"

2. **Code-Writer** (Enhance error handling)
   Prompt: "Add production-grade error handling to TenstorrentBackend.

   **Current patterns:** [Paste local-file-searcher findings]

   **Enhancements needed:**
   1. Timeout handling for bridge requests (60s default)
   2. Retry logic for transient failures (3 retries)
   3. Connection reconnection logic
   4. Informative error messages for users
   5. Proper resource cleanup on error

   **Pattern to follow:**
   ```python
   def _send_receive(self, operation: str, data: Dict, timeout: int = 60) -> Dict:
       max_retries = 3
       for attempt in range(max_retries):
           try:
               self.sock.settimeout(timeout)
               # ... operation ...
           except socket.timeout:
               logger.warning(f'Request timeout (attempt {attempt + 1}/{max_retries})')
               if attempt == max_retries - 1:
                   raise RuntimeError(f'Bridge request timed out after {timeout}s')
               self._reconnect()
           except (BrokenPipeError, ConnectionResetError) as e:
               logger.warning(f'Connection error: {e}, reconnecting...')
               self._reconnect()
   ```"

3. **Critical-Reviewer** (Review error handling)
   Prompt: "Review enhanced error handling implementation.

   Check for:
   - All exceptions properly caught
   - User-friendly error messages
   - Resource cleanup in all paths
   - Retry logic correctness
   - Logging adequacy"
```

### Task 1.3: Shared Memory Management

**Agent Sequence:**

```markdown
1. **Explore** (Understand current implementation)
   Prompt: "Explore shared memory management in TensorBridge.

   Questions to answer:
   - How are shared memory segments created?
   - How are they cleaned up?
   - What are the potential leak scenarios?
   - Are there any existing cleanup mechanisms?"

2. **Code-Writer** (Implement leak prevention)
   Prompt: "Implement robust shared memory management for TensorBridge.

   **Current implementation:** [Paste Explore findings]

   **Requirements:**
   1. Track all active shared memory segments
   2. Periodic cleanup of stale segments (5 min interval)
   3. Cleanup on abnormal termination
   4. Explicit cleanup method for graceful shutdown

   **Pattern:**
   ```python
   class TensorBridge:
       def __init__(self):
           self._active_segments = {}
           self._cleanup_interval = 300
           self._last_cleanup = time.time()

       def _periodic_cleanup(self):
           '''Clean up stale shared memory segments'''
           if time.time() - self._last_cleanup < self._cleanup_interval:
               return

           for name in list(self._active_segments.keys()):
               try:
                   shm = self._active_segments[name]
                   if not self._is_segment_active(name):
                       shm.close()
                       shm.unlink()
                       del self._active_segments[name]
               except Exception as e:
                   logger.warning(f'Cleanup failed for {name}: {e}')

           self._last_cleanup = time.time()
   ```"
```

### Task 1.4: Test Suite Development

**Agent Sequence:**

```markdown
1. **Code-Writer** (Write test suite)
   Prompt: "Create comprehensive test suite for ComfyUI-TT integration.

   **Test structure:**

   1. Unit Tests (tests/unit/)
      - test_tensor_bridge.py - Shared memory operations
      - test_backend_client.py - Socket communication
      - test_wrappers.py - ComfyUI interface compliance

   2. Integration Tests (tests/integration/)
      - test_full_denoise.py - End-to-end denoising
      - test_checkpoint_loader.py - Model initialization
      - test_error_handling.py - Error recovery

   3. Quality Tests (tests/quality/)
      - test_ssim.py - Image quality validation (SSIM >= 0.95)
      - test_determinism.py - Reproducibility with seeds

   **Key test example:**
   ```python
   def test_ssim_vs_reference():
       '''Verify SSIM >= 0.95 against PyTorch reference'''
       reference = Image.open('tests/fixtures/reference_astronaut.png')

       backend = TenstorrentBackend()
       model_id = backend.init_model('sdxl', {})
       result = backend.full_inference(
           model_id=model_id,
           prompt='An astronaut riding a horse',
           seed=42,
           steps=20,
       )
       tt_image = decode_base64_image(result['images'][0])

       ssim_score = calculate_ssim(reference, tt_image)
       assert ssim_score >= 0.95, f'SSIM {ssim_score} below threshold'
   ```

   Use pytest framework. Include fixtures for common test data."

2. **Critical-Reviewer** (Review tests)
   Prompt: "Review test suite for completeness and quality.

   Check:
   - Test coverage of critical paths
   - Edge cases covered
   - Proper fixture usage
   - Clear assertions with good error messages
   - Tests are deterministic
   - Performance tests have reasonable timeouts"
```

### Task 1.5: Caching Implementation

**Agent Sequence:**

```markdown
1. **Code-Writer** (Implement caching)
   Prompt: "Implement bridge-level caching for prompt embeddings and traces.

   **Cache layers needed:**

   1. **Prompt Embedding Cache**
   ```python
   class PromptCache:
       def __init__(self, max_size: int = 100):
           self.cache = OrderedDict()
           self.max_size = max_size

       def get_key(self, prompt: str, negative_prompt: str) -> str:
           return hashlib.md5(f'{prompt}|{negative_prompt}'.encode()).hexdigest()

       def get(self, prompt: str, negative_prompt: str) -> Optional[Dict]:
           key = self.get_key(prompt, negative_prompt)
           if key in self.cache:
               self.cache.move_to_end(key)  # LRU
               return self.cache[key]
           return None

       def set(self, prompt: str, negative_prompt: str, embeddings: Dict):
           key = self.get_key(prompt, negative_prompt)
           self.cache[key] = embeddings
           if len(self.cache) > self.max_size:
               self.cache.popitem(last=False)
   ```

   2. **Trace Cache** - Ensure compiled traces are cached by step count

   3. **Configuration** - Add cache size limits and TTL settings

   Integrate caching into bridge server's full_inference handler."

2. **Critical-Reviewer** (Review caching)
   Prompt: "Review caching implementation.

   Check:
   - Cache invalidation strategy is sound
   - Memory limits enforced
   - Thread-safety if needed
   - Cache hit/miss logging for debugging
   - Performance improvement measurable"
```

### Task 1.6: Documentation (Week 5)

**Agent Sequence:**

```markdown
1. **Communications-Translator** (Create user docs)
   Prompt: "Create user-facing documentation for ComfyUI-TT integration v1.0.

   **Content to transform into documentation:**
   - Technical specs from implementation plan
   - Installation and setup procedures
   - Node usage examples
   - Troubleshooting common issues

   **Documents to create:**

   1. **Quick Start Guide (docs/quickstart.md)**
      - 5-minute setup instructions
      - First workflow walkthrough
      - Expected output

   2. **User Reference (docs/user_guide.md)**
      - Node reference (TT_CheckpointLoader, TT_FullDenoise, etc.)
      - Configuration options explained in plain language
      - Performance tuning tips
      - Best practices

   3. **Troubleshooting Guide (docs/troubleshooting.md)**
      - Common issues and solutions
      - How to interpret logs
      - When to file a bug report

   **Tone:** Clear, accessible, assumes user familiar with ComfyUI but new to TT backend."

2. **Critical-Reviewer** (Review documentation)
   Prompt: "Review user documentation for clarity and completeness.

   Check:
   - Instructions are clear and actionable
   - No assumed knowledge gaps
   - Examples work as written
   - Troubleshooting covers actual user issues
   - Grammar and formatting consistent"
```

---

## Phase 2: Native Hook Integration (Weeks 6-11)

### Task 2.1: TTModelWrapper Refactoring

**Agent Sequence:**

```markdown
1. **Explore** (Understand model_function_wrapper)
   Prompt: "Explore ComfyUI's model_function_wrapper hook system.

   **Files to investigate:**
   - /home/tt-admin/ComfyUI-tt_standalone/comfy/samplers.py
   - /home/tt-admin/ComfyUI-tt_standalone/comfy/model_patcher.py

   **Questions:**
   1. How does model_function_wrapper get called?
   2. What parameters does it receive?
   3. What is it expected to return?
   4. How do ControlNet and IP-Adapter use this hook?
   5. What are the state management requirements?"

2. **Code-Writer** (Refactor TTModelWrapper)
   Prompt: "Refactor TTModelWrapper for hook compatibility based on exploration findings.

   **Context:** [Paste Explore findings]

   **Requirements:**

   1. **State Management**
   ```python
   class TTModelWrapper:
       def __init__(self, ...):
           self._execution_context = {
               'current_step': 0,
               'total_steps': 0,
               'accumulated_control': None,
               'ip_adapter_embeds': None,
           }

       def prepare_for_sampling(self, num_steps: int):
           '''Called before sampling loop starts'''
           self._execution_context['total_steps'] = num_steps
           self._execution_context['current_step'] = 0
           self.backend._send_receive('reset_scheduler', {
               'model_id': self.model_id,
               'num_steps': num_steps,
           })
   ```

   2. **Hook-Compatible apply_model**
   ```python
   def apply_model(self, x, t, c_concat=None, c_crossattn=None,
                   control=None, transformer_options=None, **kwargs):
       '''Apply model for single denoising step with hook support'''

       # 1. Apply ControlNet conditioning if present
       if control is not None:
           x, control_residuals = self._apply_controlnet(x, control)
       else:
           control_residuals = None

       # 2. Apply IP-Adapter if present
       if 'ip_adapter' in (transformer_options or {}):
           c_crossattn = self._apply_ip_adapter(c_crossattn, transformer_options)

       # 3. Call bridge for UNet execution
       epsilon = self._bridge_unet_call(x, t, c_crossattn, control_residuals)

       # 4. Convert epsilon to denoised
       denoised = x - epsilon * t

       # 5. Increment step counter
       self._execution_context['current_step'] += 1

       return denoised
   ```

   File: /home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/wrappers.py"

3. **Critical-Reviewer** (Review refactoring)
   Prompt: "Review TTModelWrapper refactoring for hook compatibility.

   Check:
   - State management is thread-safe
   - Hook interface matches ComfyUI expectations
   - Backward compatibility with TT_FullDenoise
   - Error handling for partial implementations
   - Performance impact is minimal"
```

### Task 2.2: ControlNet Integration

**Agent Sequence:**

```markdown
1. **Explore** (Understand ControlNet flow)
   Prompt: "Explore ControlNet implementation in ComfyUI.

   **Files:**
   - /home/tt-admin/ComfyUI-tt_standalone/comfy/controlnet.py
   - /home/tt-admin/ComfyUI-tt_standalone/comfy/cldm/cldm.py

   **Questions:**
   1. How are ControlNet residuals computed?
   2. At what layers are residuals applied?
   3. What format are residuals passed to UNet?
   4. How does strength parameter work?
   5. Can we run ControlNet on CPU and pass residuals to TT UNet?"

2. **Code-Writer** (Implement ControlNet support)
   Prompt: "Implement ControlNet integration for TTModelWrapper.

   **Context:** [Paste Explore findings]

   **Strategy:** Hybrid execution
   - ControlNet runs on CPU (or TT if available)
   - Residuals passed to TT UNet via bridge

   **Implementation:**

   1. Add ControlNet handler:
   ```python
   class ControlNetHandler:
       def __init__(self, backend):
           self.backend = backend

       def compute_residuals(self, control_model, x, hint, timestep, c):
           '''Compute ControlNet residuals'''
           # Run ControlNet (CPU)
           with torch.no_grad():
               residuals = control_model(x, hint, timestep, c)

           return residuals

       def prepare_for_bridge(self, residuals):
           '''Prepare residuals for bridge transfer'''
           return {
               f'control_{i}': self.backend.tensor_bridge.tensor_to_shm(r)
               for i, r in enumerate(residuals)
           }
   ```

   2. Update bridge server to accept control residuals

   3. Add TT_ControlNet node (optional, for TT-native ControlNet)"

3. **Problem-Investigator** (Test and debug)
   Prompt: "Test ControlNet integration and debug any issues.

   **Test cases:**
   1. Canny edge ControlNet
   2. Depth ControlNet
   3. OpenPose ControlNet
   4. Multiple ControlNets combined

   **For each test:**
   - Generate image with ControlNet
   - Compare quality vs CPU reference
   - Measure performance (target: < 4.5s)
   - Check for artifacts or quality issues

   Debug any issues found and iterate until working."

4. **Critical-Reviewer** (Review ControlNet)
   Prompt: "Review ControlNet integration implementation.

   Check:
   - Correctness of residual application
   - Performance meets targets
   - Quality matches reference
   - Multiple ControlNets work
   - Strength parameter works correctly"
```

### Task 2.3: IP-Adapter Integration

**Agent Sequence:**

```markdown
1. **Explore** (Understand IP-Adapter)
   Prompt: "Explore IP-Adapter implementation in ComfyUI.

   **Files:**
   - Search for 'ip_adapter' or 'ipadapter' in ComfyUI-tt_standalone

   **Questions:**
   1. How does IP-Adapter modify cross-attention?
   2. How are reference images encoded?
   3. What are the integration points in UNet?
   4. Can we run image encoder on CPU?
   5. What data needs to pass to TT UNet?"

2. **Code-Writer** (Implement IP-Adapter)
   Prompt: "Implement IP-Adapter integration for TTModelWrapper.

   **Context:** [Paste Explore findings]

   **Strategy:**
   - Image encoder runs on CPU
   - Image embeddings passed to TT UNet
   - TT UNet modifies cross-attention with IP-Adapter

   **Implementation:**
   ```python
   class IPAdapterHandler:
       def __init__(self, backend):
           self.backend = backend
           self.image_encoder = self._load_image_encoder()  # CPU

       def encode_reference_image(self, image: torch.Tensor) -> torch.Tensor:
           '''Encode reference image to embeddings'''
           with torch.no_grad():
               embeddings = self.image_encoder(image)
           return embeddings

       def prepare_for_unet(self, embeddings: torch.Tensor) -> Dict:
           '''Prepare embeddings for bridge transfer'''
           return {
               'ip_adapter_embeds': self.backend.tensor_bridge.tensor_to_shm(embeddings)
           }
   ```

   Update bridge server to handle IP-Adapter embeddings."

3. **Problem-Investigator** (Test IP-Adapter)
   Prompt: "Test IP-Adapter integration.

   **Tests:**
   1. Single reference image
   2. Multiple reference images
   3. IP-Adapter + ControlNet combined
   4. Variable IP-Adapter strength

   Measure quality and performance. Target: < 5.0s per image."
```

### Task 2.4: Integration Testing (Week 10)

**Agent Sequence:**

```markdown
1. **Integration-Orchestrator** (Coordinate complex workflow testing)
   Prompt: "Orchestrate comprehensive integration testing of Phase 2 features.

   **Test Matrix:**

   | Feature 1 | Feature 2 | Feature 3 | Expected Time | Priority |
   |-----------|-----------|-----------|---------------|----------|
   | Basic     | -         | -         | 2.8s          | P0       |
   | ControlNet| -         | -         | 4.5s          | P0       |
   | IP-Adapter| -         | -         | 5.0s          | P0       |
   | ControlNet| IP-Adapter| -         | 6.5s          | P1       |
   | ControlNet| IP-Adapter| LoRA      | 7.0s          | P1       |

   **Workflow tests:**
   1. Basic txt2img with TT backend
   2. txt2img + Canny ControlNet
   3. txt2img + IP-Adapter
   4. Complex: ControlNet + IP-Adapter + custom sampler
   5. Edge cases: high guidance, low steps, non-square

   **Orchestration:**
   - Spawn problem-investigator for any failures
   - Spawn code-writer for fixes
   - Run critical-reviewer on fixes
   - Repeat until all tests pass

   **Deliverable:**
   - Test results matrix
   - Performance benchmark report
   - List of issues found and fixed
   - Go/no-go for v2.0 release"

2. **Critical-Reviewer** (Final Phase 2 review)
   Prompt: "Perform final review before v2.0 release.

   **Review scope:**
   - All Phase 2 code changes
   - Test coverage adequacy
   - Documentation completeness
   - Performance vs targets
   - Security considerations
   - Backward compatibility

   **Approval criteria:**
   - All P0 tests passing
   - Performance targets met
   - No critical security issues
   - Documentation complete"
```

---

## Phase 3: Optimization & Production (Weeks 12-14)

### Task 3.1: Performance Profiling

**Agent Sequence:**

```markdown
1. **Problem-Investigator** (Profile and identify bottlenecks)
   Prompt: "Profile ComfyUI-TT integration end-to-end and identify bottlenecks.

   **Profiling methodology:**
   1. Use Python cProfile for high-level hotspots
   2. Use ttnn profiler for device-level analysis
   3. Measure per-component latencies
   4. Identify optimization opportunities

   **Target breakdown (20 steps, basic txt2img):**
   ```
   Total time: ~2.8s target
   ------------------------------------------
   Text encoding:     ~0.4s  (14%)
   UNet (20 steps):   ~2.0s  (71%)  <-- Primary target
   VAE decode:        ~0.3s  (11%)
   Overhead (IPC):    ~0.1s  (4%)
   ```

   **Deliverable:**
   - Detailed profiling report
   - Top 5 bottlenecks identified
   - Optimization recommendations with estimated impact"

2. **Code-Writer** (Implement optimizations)
   Prompt: "Implement performance optimizations based on profiling.

   **Context:** [Paste problem-investigator findings]

   **Optimizations to implement:**
   1. Trace execution optimization
   2. Reduce host-device synchronization
   3. Use pinned memory for transfers
   4. Batch small IPC messages
   5. Pre-warm trace for common step counts

   Target: Meet performance goals from plan."

3. **Critical-Reviewer** (Review optimizations)
   Prompt: "Review performance optimizations.

   Check:
   - Optimizations are safe (no correctness issues)
   - Measurable performance improvement
   - No negative side effects
   - Code remains maintainable"
```

### Task 3.2: Load Testing

**Agent Sequence:**

```markdown
1. **Code-Writer** (Create load test suite)
   Prompt: "Create comprehensive load testing suite.

   **Test scenarios:**

   1. **Sustained Load**: 100 requests over 1 hour
   ```python
   def test_sustained_load():
       results = []
       for i in range(100):
           start = time.time()
           result = backend.full_inference(...)
           duration = time.time() - start
           results.append({'request': i, 'duration': duration, 'success': True})

       success_rate = sum(1 for r in results if r['success']) / len(results)
       assert success_rate >= 0.999, f'Success rate {success_rate} < 99.9%'
   ```

   2. **Burst Load**: 10 concurrent requests
   3. **Extended Run**: 24-hour continuous operation
   4. **Memory Stress**: Generate until OOM, verify recovery

   Use pytest and add memory profiling."

2. **Problem-Investigator** (Run and analyze load tests)
   Prompt: "Run load tests and investigate any failures.

   Execute all load test scenarios and collect:
   - Success rate (target: 99.9%)
   - Memory usage over time (should be flat, no leaks)
   - Error logs for any failures
   - Recovery time after failures

   Investigate and fix any issues found."
```

### Task 3.3: Final Documentation

**Agent Sequence:**

```markdown
1. **Communications-Translator** (Production deployment guide)
   Prompt: "Create production deployment guide for ComfyUI-TT.

   **Content to cover:**
   1. System requirements (hardware, OS, dependencies)
   2. Installation steps (detailed, tested)
   3. Configuration options (with explanations)
   4. Monitoring and logging setup
   5. Backup and recovery procedures
   6. Scaling considerations
   7. Troubleshooting production issues

   **Audience:** System administrators and DevOps engineers

   **Tone:** Professional, precise, assumes technical knowledge but not specific to ComfyUI/TT"

2. **Integration-Orchestrator** (Consolidate all docs)
   Prompt: "Consolidate and organize all documentation for final release.

   **Documentation structure:**
   ```
   docs/
   ├── README.md (overview and quick links)
   ├── quickstart.md (5-minute start)
   ├── user_guide.md (comprehensive user docs)
   ├── developer_guide.md (for contributors)
   ├── api_reference.md (API docs)
   ├── architecture.md (system design)
   ├── troubleshooting.md (common issues)
   ├── deployment.md (production guide)
   └── changelog.md (version history)
   ```

   Ensure:
   - Consistent formatting
   - Working internal links
   - Up-to-date code examples
   - Complete coverage of features"

3. **Critical-Reviewer** (Final doc review)
   Prompt: "Final documentation review before release.

   Check:
   - Accuracy (test all code examples)
   - Completeness (all features documented)
   - Clarity (understandable by target audience)
   - Consistency (formatting, terminology)
   - Links work"
```

---

## Execution Strategy

### Parallel Agent Deployment

**Week 1 (Phase 0) - Run in Parallel:**
```bash
# Launch both investigations simultaneously
Agent 1: problem-investigator (timestep bug)
Agent 2: problem-investigator (CFG batching)

# Once both complete:
Agent 3: integration-orchestrator (go/no-go decision)
```

**Week 2-4 - Multiple Streams:**
```bash
# Stream 1: Core bug fixes (sequential)
Agent A: code-writer (fixes) -> critical-reviewer (review)

# Stream 2: Error handling (parallel)
Agent B: code-writer (error handling)

# Stream 3: Shared memory (parallel)
Agent C: code-writer (shared memory)

# Stream 4: Tests (parallel, after fixes)
Agent D: code-writer (test suite)
```

**Week 7-9 - Feature Integration (Parallel):**
```bash
# All can run simultaneously:
Agent A: code-writer (ControlNet)
Agent B: code-writer (IP-Adapter)
Agent C: code-writer (Sampler compat)

# Then sequential review:
Agent D: critical-reviewer (review all)
```

### Agent Communication Pattern

```markdown
When Agent A completes work that Agent B depends on:

1. Agent A produces deliverable (code, report, findings)
2. You (orchestrator) extract key information
3. Pass to Agent B via prompt: "Context from Agent A: [summary]"
4. Agent B uses context to proceed

Example:
- problem-investigator finds root cause
- You extract: "Root cause is X in file Y, fix is Z"
- code-writer receives: "Implement fix Z in file Y to address X"
```

---

## Success Metrics Tracking

After each phase, verify:

**Phase 0:**
- [ ] Timestep investigation complete with findings
- [ ] CFG batching verified as working/fixed
- [ ] SSIM >= 0.90 achieved
- [ ] Go/no-go decision documented

**Phase 1:**
- [ ] All critical bugs fixed
- [ ] SSIM >= 0.95 achieved
- [ ] Test suite with 50+ tests passing
- [ ] Documentation complete
- [ ] v1.0 released

**Phase 2:**
- [ ] ControlNet working (< 4.5s)
- [ ] IP-Adapter working (< 5.0s)
- [ ] All standard samplers compatible
- [ ] Integration tests passing
- [ ] v2.0 released

**Phase 3:**
- [ ] Performance targets met
- [ ] Load tests passed (99.9% uptime)
- [ ] Documentation complete
- [ ] Production release

---

## Rollback / Pivot Criteria

**Stop and reassess if:**

1. **Phase 0:** Either critical validation fails and can't be fixed in 3 days
   - **Action:** Pivot to bridge-only mode (TT_FullDenoise only)

2. **Phase 1:** SSIM can't reach 0.95 after investigation
   - **Action:** Accept current SSIM if > 0.90, document limitations

3. **Phase 2:** Per-step performance > 50% slower than bridge loop
   - **Action:** Keep bridge loop as primary, hooks as experimental

4. **Phase 3:** Load testing shows < 95% success rate
   - **Action:** Delay release, continue hardening

---

## Agent Usage Summary

| Agent Type | Primary Use Cases | Phase |
|------------|-------------------|-------|
| **problem-investigator** | Phase 0 validation, debugging, profiling | 0, 1, 2, 3 |
| **code-writer** | All implementation tasks | 1, 2, 3 |
| **critical-reviewer** | Code review, security, quality checks | 1, 2, 3 |
| **explore** | Understanding architecture, finding code | 1, 2 |
| **local-file-searcher** | Quick lookups, finding references | 1, 2 |
| **integration-orchestrator** | Complex workflows, multiple agent coordination | 0, 2, 3 |
| **communications-translator** | User documentation creation | 1, 3 |

---

## Next Steps

To begin execution:

1. **Review and approve this execution prompt**
2. **Start with Phase 0, Task 0.1** (Timestep investigation)
3. **Launch problem-investigator agent** with Task 0.1 prompt
4. **Based on results, proceed to Task 0.2** (CFG batching)
5. **After both validations, launch integration-orchestrator** for go/no-go

**First command to run:**
```markdown
Launch problem-investigator agent with Task 0.1 prompt from this document.
```

---

*This execution prompt is designed to maximize parallel agent usage while maintaining clear dependencies and quality gates. Each task has specific prompts ready to paste to the appropriate agent.*
