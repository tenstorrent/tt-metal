# Code Review Checklist

**Project:** Phase 1 Bridge Extension  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md

---

## Review Information

**PR/Change ID:** _______________  
**Author:** _______________  
**Reviewer:** _______________  
**Date:** _______________  
**Files Changed:** _______________

---

## 1. Specification Compliance

### Does Code Implement Spec from Prompt?

- [ ] Function signature matches specification
- [ ] Input parameters match documented types
- [ ] Output format matches specification
- [ ] Error handling matches specified behavior
- [ ] Edge cases from spec are handled

**Spec Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md, Section: ___

**Notes:**
```
[Reviewer notes on spec compliance]
```

---

## 2. Model-Agnostic Handling

### Is Model-Agnostic Handling Correct?

- [ ] Uses MODEL_CONFIGS for channel counts
- [ ] No hardcoded model-specific values
- [ ] Model type determined from config/params
- [ ] Future models can be added via config only
- [ ] Error messages include model type context

**Check for these patterns:**

```python
# BAD: Hardcoded values
if C != 4:  # SDXL specific
    raise ValueError(...)

# GOOD: Config-based
config = MODEL_CONFIGS[model_type]
if C != config["latent_channels"]:
    raise ValueError(f"Expected {config['latent_channels']} for {model_type}, got {C}")
```

**Notes:**
```
[Reviewer notes on model-agnostic handling]
```

---

## 3. Reusability Comments

### Are Reusability Comments Present?

- [ ] Major functions have "Reusability Note for Native Integration"
- [ ] Comments explain what's portable
- [ ] Comments explain what needs modification
- [ ] Comments explain rationale for design

**Required format:**

```python
def handle_denoise_step_single(self, params):
    """
    Execute a single denoising step.
    
    Reusability Note for Native Integration:
        This operation implements the per-timestep pattern required by ComfyUI.
        The same pattern will be used in native integration - only IPC layer changes.
        
        Portable: Parameter handling, format conversion, session tracking
        Modify: Transport layer (socket -> direct call)
        Rationale: Per-step enables ControlNet/IP-Adapter ecosystem
    """
```

**Notes:**
```
[Reviewer notes on reusability comments]
```

---

## 4. Error Handling

### Are Error Cases Handled?

- [ ] All required parameters validated
- [ ] Type checking for inputs
- [ ] Range validation where applicable
- [ ] Graceful handling of unexpected inputs
- [ ] Error messages are clear and actionable
- [ ] Recovery suggestions provided where possible
- [ ] Errors logged appropriately

**Check for these patterns:**

```python
# GOOD: Comprehensive error handling
def handle_denoise_step_single(self, params):
    # Validate required parameters
    if "session_id" not in params:
        return self._error_response(
            "MissingParameter",
            "session_id is required",
            suggestion="Create a session first with session_create"
        )
    
    # Type validation
    if not isinstance(params["timestep"], (int, float)):
        return self._error_response(
            "InvalidType",
            f"timestep must be numeric, got {type(params['timestep'])}",
            recoverable=True
        )
```

**Notes:**
```
[Reviewer notes on error handling]
```

---

## 5. Test Coverage

### Are Tests Comprehensive?

- [ ] Unit tests for new functions
- [ ] Unit tests for edge cases
- [ ] Integration tests for workflows
- [ ] Regression tests (existing functionality not broken)
- [ ] Performance tests (if performance-critical)

**Test coverage checklist:**

| Function/Component | Unit Tests | Edge Cases | Integration | Notes |
|-------------------|------------|------------|-------------|-------|
| handle_denoise_step_single | [ ] | [ ] | [ ] | |
| SessionManager | [ ] | [ ] | [ ] | |
| Model config lookup | [ ] | [ ] | [ ] | |
| Format conversion | [ ] | [ ] | [ ] | |

**Minimum requirements:**
- New functions: At least 3 unit tests each
- Edge cases: Null/empty inputs, boundary values
- Error paths: At least 1 test per error type

**Notes:**
```
[Reviewer notes on test coverage]
```

---

## 6. Performance

### Is Performance Acceptable?

- [ ] No obvious performance issues
- [ ] Avoids unnecessary allocations in hot path
- [ ] Uses appropriate data structures
- [ ] Caching used where beneficial
- [ ] No blocking operations in critical path

**Performance considerations:**

```python
# BAD: Allocating in hot path
def handle_step(self, params):
    config = MODEL_CONFIGS.copy()  # Unnecessary allocation
    
# GOOD: Reference existing
def handle_step(self, params):
    config = MODEL_CONFIGS[self.model_type]  # Direct lookup

# BAD: Repeated computation
for step in range(20):
    expected_channels = MODEL_CONFIGS[model_type]["latent_channels"]  # Redundant lookup
    
# GOOD: Compute once
expected_channels = MODEL_CONFIGS[model_type]["latent_channels"]
for step in range(20):
    ...
```

**Benchmark results (if applicable):**
| Operation | Baseline | With Changes | Acceptable? |
|-----------|----------|--------------|-------------|
| | | | [ ] Yes [ ] No |

**Notes:**
```
[Reviewer notes on performance]
```

---

## 7. ADR References

### Are ADR References Correct?

- [ ] Design decisions reference appropriate ADR
- [ ] ADR number/title is correct
- [ ] Code follows ADR recommendations
- [ ] Deviations from ADR are documented and justified

**Expected ADR references:**

| Decision Area | Expected ADR | Referenced? |
|--------------|--------------|-------------|
| Per-timestep pattern | ADR-001 | [ ] |
| Scheduler synchronization | ADR-002 | [ ] |
| ControlNet integration | ADR-003 | [ ] |

**Notes:**
```
[Reviewer notes on ADR references]
```

---

## 8. Code Quality

### General Quality Checks

- [ ] Code follows project style guidelines
- [ ] Variable names are descriptive
- [ ] Functions are appropriately sized (< 50 lines preferred)
- [ ] No dead code or commented-out code
- [ ] Imports are organized
- [ ] No security issues (secrets, injection, etc.)

### Documentation

- [ ] All public functions have docstrings
- [ ] Complex logic has inline comments
- [ ] TODO items have associated ticket/issue
- [ ] README updated if needed

### Maintainability

- [ ] Code is DRY (Don't Repeat Yourself)
- [ ] Single Responsibility Principle followed
- [ ] Dependencies are explicit
- [ ] Configuration is externalized

**Notes:**
```
[Reviewer notes on code quality]
```

---

## 9. Security Considerations

- [ ] No sensitive data logged
- [ ] Input validation prevents injection
- [ ] File paths validated/sanitized
- [ ] No hardcoded credentials
- [ ] External inputs sanitized

**Notes:**
```
[Reviewer notes on security]
```

---

## 10. Backward Compatibility

- [ ] Existing APIs not broken
- [ ] New parameters have defaults
- [ ] Deprecation warnings for changed behavior
- [ ] Migration path documented if breaking

**Notes:**
```
[Reviewer notes on compatibility]
```

---

## Review Summary

### Overall Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Spec compliance | [ ] Pass [ ] Needs Work [ ] Fail | |
| Model-agnostic | [ ] Pass [ ] Needs Work [ ] Fail | |
| Reusability comments | [ ] Pass [ ] Needs Work [ ] Fail | |
| Error handling | [ ] Pass [ ] Needs Work [ ] Fail | |
| Test coverage | [ ] Pass [ ] Needs Work [ ] Fail | |
| Performance | [ ] Pass [ ] Needs Work [ ] Fail | |
| ADR references | [ ] Pass [ ] Needs Work [ ] Fail | |
| Code quality | [ ] Pass [ ] Needs Work [ ] Fail | |

### Required Changes

| Priority | Change Required | Location |
|----------|-----------------|----------|
| [ ] Blocking | | |
| [ ] High | | |
| [ ] Medium | | |
| [ ] Low | | |

### Approval

**Review Decision:** [ ] APPROVED  [ ] APPROVED WITH COMMENTS  [ ] CHANGES REQUESTED  [ ] REJECTED

**Reviewer Signature:** _______________  
**Date:** _______________

---

**Template Version:** 1.0  
**Created:** December 16, 2025
