# Quick Start Guide

Get started with automated fix implementation in 3 steps.

## Prerequisites

- Git repository access
- GitHub CLI (`gh`) installed and authenticated
- Python 3.8+
- Access to development environment with test hardware (if needed)

## Step 1: Configure Your Task (1 minute)

Edit `info.json` in this directory:

```bash
cd .github/scripts/codebase-improvements
nano info.json  # or your preferred editor
```

**Template:**
```json
{
  "deterministic": true,
  "url": "",
  "prompt": "Your detailed description of what to fix",
  "raw-logs": "",
  "existing-test-path": ""
}
```

**Fill in:**
- `deterministic`: `true` if always fails, `false` if intermittent
- `url`: GitHub Actions run URL (or leave empty)
- `prompt`: Clear description of the issue and desired fix
- `raw-logs`: Paste logs here if not using URL (or leave empty)
- `existing-test-path`: Path to existing test file to skip reproduction (or leave empty)

**Important:** Use either `existing-test-path` OR (`url` OR `raw-logs`), not multiple.

## Step 2: Run the Automation (20 minutes)

```bash
./run.sh
```

The script will pause at key points and prompt you to invoke Claude:

1. **First Pause**: Create reproduction test
   - Read the prompt shown in terminal
   - Invoke Claude with the reproduction AI_PROMPT.md
   - Verify test reproduces the failure
   - Press ENTER to continue

2. **Second Pause**: Implement fix
   - Read the prompt shown in terminal
   - Invoke Claude with the implementation AI_PROMPT.md
   - Let Claude analyze, fix, and create PR
   - Press ENTER to continue

## Step 3: Review Results

Check the generated report:

```bash
ls -lt outputs/  # Find latest report
cat outputs/2026-02-04_16-30-00_<description>.md
```

The report contains:
- ✅ Success/failure status
- 🔗 PR link (if created)
- 📝 Detailed execution log
- 👥 Recommended developers to contact
- 📊 Performance metrics (if applicable)

## Example: Fix Existing Timeout Issue (Using Existing Test)

We have an existing test for the wide tensor timeout. Here's how to fix it directly:

```bash
# 1. Configure (using existing test - skips reproduction phase!)
cat > info.json <<'EOF'
{
  "deterministic": true,
  "url": "",
  "prompt": "Fix ttnn.to_torch() timeout with wide tensors [1, 151936]. Optimize completion queue copy in system_memory_manager.cpp",
  "raw-logs": "",
  "existing-test-path": "reproduce-deterministic-failures/timeout-in-datamovement/tests/test_gather_timeout_stress.py"
}
EOF

# 2. Run
./run.sh

# 3. Script will skip reproduction and go directly to fix implementation

# 4. When prompted, invoke Claude for fix implementation
# Claude will analyze the test, optimize the copy operation, and create PR

# 5. Review the PR created by Claude
gh pr view <PR-number>
```

This approach is faster when you already have a working reproduction test!

## Common Scenarios

### Scenario 1: I Have a CI Failure URL

```json
{
  "deterministic": true,
  "url": "https://github.com/tenstorrent/tt-metal/actions/runs/12345678",
  "prompt": "Test XYZ fails with error ABC. Fix the root cause.",
  "raw-logs": ""
}
```

### Scenario 2: I Have Raw Logs

```json
{
  "deterministic": false,
  "url": "",
  "prompt": "Intermittent hang in reduce_scatter. Fix race condition.",
  "raw-logs": "... paste full error logs here ..."
}
```

### Scenario 3: I Have an Existing Test

```json
{
  "deterministic": true,
  "url": "",
  "prompt": "Fix the timeout issue demonstrated by the existing test.",
  "raw-logs": "",
  "existing-test-path": "path/to/test_file.py"
}
```

### Scenario 4: I Want to Optimize Performance

```json
{
  "deterministic": true,
  "url": "https://github.com/tenstorrent/tt-metal/actions/runs/12345678",
  "prompt": "Llama70B prefill is 450ms. Reduce to <400ms by optimizing attention.",
  "raw-logs": "",
  "existing-test-path": ""
}
```

## What Happens During Execution

```
┌─────────────────────────────────────────┐
│ 1. Parse info.json                      │
│    - Read configuration                 │
│    - Validate inputs                    │
│    - Check for existing-test-path       │
└─────────────────────────────────────────┘
           ↓
      ┌────┴────┐
      │ Existing │
      │  test?   │
      └────┬────┘
      Yes ↓    ↓ No
          │    │
          │    ┌─────────────────────────────────────────┐
          │    │ 2. Fetch/Prepare Logs                   │
          │    │    - Download from URL or use raw logs │
          │    │    - Save to temporary directory        │
          │    └─────────────────────────────────────────┘
          │               ↓
          │    ┌─────────────────────────────────────────┐
          │    │ 3. Create Reproduction Test (5 min)     │
          │    │    ⏸  Manual: Invoke Claude             │
          │    │    - AI reads logs and original test    │
          │    │    - Creates minimal reproduction       │
          │    │    - Verifies it fails as expected      │
          │    │    - Commits test to current branch     │
          │    └─────────────────────────────────────────┘
          │               ↓
          └───────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ 4. Implement Fix (15 min)               │
│    - Creates fix branch from main       │
│    - Copies test to new branch          │
│    ⏸  Manual: Invoke Claude             │
│    - AI analyzes root cause             │
│    - Implements fix iteratively         │
│    - Verifies test passes (5x)          │
│    - Creates draft PR                   │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ 5. Generate Report                      │
│    - Writes markdown report             │
│    - Includes PR link                   │
│    - Lists next steps                   │
│    - Names relevant developers          │
└─────────────────────────────────────────┘
```

## Time Expectations

| Phase | Time | Cumulative |
|-------|------|------------|
| Configuration | 1 min | 1 min |
| Log fetching | 1 min | 2 min |
| Reproduction test | 5 min | 7 min |
| Fix implementation | 15 min | 22 min |
| Report generation | 1 min | 23 min |
| **Total** | **23 min** | - |

## Success Indicators

✅ **Full Success**
- Reproduction test created and fails as expected
- Fix implemented and test passes reliably
- Draft PR created with clear description
- Report shows "Status: ✅ Success"

⚠️ **Partial Success**
- Reproduction works but fix incomplete
- Test passes sometimes but not always
- Improvement made but target not met
- Report shows "Status: ⚠️ Partial"

❌ **Failure**
- Cannot reproduce the issue
- Cannot identify root cause
- Fix doesn't work after multiple attempts
- Report shows "Status: ❌ Failed"

**Even failures are valuable** - they document what was tried and recommend next steps.

## Troubleshooting

### Issue: Script says "info.json not found"

**Solution:**
```bash
# Make sure you're in the right directory
cd .github/scripts/codebase-improvements
ls info.json  # Should exist
```

### Issue: Claude cannot reproduce the failure

**Possible causes:**
- Logs don't contain enough information
- Environment differs from CI
- Issue is hardware-specific

**Solution:**
- Provide more detailed logs
- Include environment variables from CI
- Try running on same hardware as CI

### Issue: Fix doesn't work

**Possible causes:**
- Root cause misidentified
- Approach is incorrect
- Issue is more complex than expected

**Solution:**
- Review AI's hypothesis in report
- Provide more context in prompt
- Break into smaller tasks
- Consult experts listed in report

### Issue: Script hangs

**Solution:**
- Check if you need to press ENTER at manual step
- Look for prompt requiring Claude invocation
- Check terminal for error messages

## Next Steps After Success

1. **Review the PR**
   ```bash
   gh pr view <PR-number>
   gh pr diff <PR-number>
   ```

2. **Run recommended CI workflows**
   - Check PR description for list
   - Trigger workflows through GitHub UI
   - Wait for results

3. **Request review**
   ```bash
   gh pr ready <PR-number>  # Mark as ready for review
   # Tag reviewers mentioned in report
   ```

4. **Monitor and merge**
   - Address review comments
   - Verify CI passes
   - Merge when approved

## Learn More

- **Full documentation**: `README.md`
- **Detailed examples**: `EXAMPLE_usage.md`
- **Reproduction workflows**: `reproduce-*/README.md`
- **Fix implementation**: `implementing-features/README.md`

## Getting Help

If automation fails or you need assistance:

1. **Check the report** in `outputs/` for detailed diagnostics
2. **Review logs** saved during execution
3. **Consult listed developers** in the report
4. **File an issue** describing what happened

## Tips for Success

💡 **Write clear prompts** - The more specific, the better
💡 **Include context** - Explain what should happen vs what does
💡 **Use CI URLs when possible** - More complete than raw logs
💡 **Review AI decisions** - Don't blindly accept changes
💡 **Iterate if needed** - First attempt may not be perfect

---

**Ready to start?** Edit `info.json` and run `./run.sh`!
