# Block Variants Automation Scripts

This directory contains scripts to automate the implementation of block variants for the tt-metal Compute API.

## Quick Start

```bash
# Run the full automated implementation
./run_agent_implementation.sh

# Or run a specific phase
./run_agent_implementation.sh --phase 1

# Dry run (no changes)
./run_agent_implementation.sh --dry-run
```

## Scripts

### 1. `run_agent_implementation.sh` (Main Entry Point)

**Purpose**: Bash wrapper that sets up environment and runs the Python implementation script.

**Features**:
- Sources `~/.bashrc` for API configuration
- Validates prerequisites (Python, API keys, repository)
- Provides colored output and progress indicators
- Handles errors gracefully

**Usage**:
```bash
./run_agent_implementation.sh [OPTIONS]

Options:
  --phase N       Run only phase N (1-7)
  --dry-run       Show what would be done without making changes
  --verbose       Show detailed output
  --skip-api      Skip API calls, use cached results
  --help          Show help message
```

**Examples**:
```bash
# Run all phases
./run_agent_implementation.sh

# Run only inventory phase
./run_agent_implementation.sh --phase 1

# Dry run with verbose output
./run_agent_implementation.sh --dry-run --verbose

# Run phase 2 (template generation)
./run_agent_implementation.sh --phase 2
```

### 2. `add_block_variants.py` (Core Implementation)

**Purpose**: Python script implementing the 7-phase agent plan.

**Architecture**:
- `AIAgent` class: Handles LLM API calls (Anthropic/OpenAI)
- `BlockVariantsImplementation` class: Implements each phase
- Phase-based execution with caching

**Direct Usage** (if you don't want the bash wrapper):
```bash
python3 add_block_variants.py --phase 1 --verbose
```

## The 7 Phases

### Phase 1: Inventory
**Agent**: Discovery Agent
**Purpose**: Find existing tile operations and identify missing block variants
**Output**: `cache/phase_1.json`
**Actions**:
- Scans `tt_metal/include/compute_kernel_api/*.h`
- Finds `*_tile()` and `*_tiles()` functions
- Checks for existing `*_block()` functions
- Creates inventory of missing block variants

**Run Standalone**:
```bash
./run_agent_implementation.sh --phase 1
```

### Phase 2: Template Generation
**Agent**: Code Generator Agent
**Purpose**: Generate C++ templates for each missing block variant
**Output**: `cache/phase_2.json`
**Actions**:
- Creates block variant templates following for-loop pattern
- Adds compile-time template parameters (`Ht`, `Wt`)
- Adds `static_assert` for DEST capacity checks
- Includes comprehensive doc comments

**Run Standalone**:
```bash
./run_agent_implementation.sh --phase 2
```

**Template Pattern**:
```cpp
template <uint32_t Ht, uint32_t Wt>
ALWI void add_block(...) {
    static_assert(Ht * Wt <= 16);
    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t w = 0; w < Wt; w++) {
            add_tiles(...);  // Call existing function
        }
    }
}
```

### Phase 3: Code Integration
**Agent**: Integration Agent
**Purpose**: Insert generated templates into header files
**Output**: `cache/phase_3.json`
**Actions**:
- Locates insertion points in header files
- Inserts templates after corresponding tile functions
- Maintains file structure and formatting
- Validates syntax after each insertion

**Run Standalone**:
```bash
./run_agent_implementation.sh --phase 3
```

### Phase 4: Documentation
**Agent**: Documentation Agent
**Purpose**: Generate API reference documentation
**Output**: `BLOCK_VARIANTS_API.md`, `cache/phase_4.json`
**Actions**:
- Creates API reference for each new function
- Generates usage examples
- Updates TASK.md with completion status

**Run Standalone**:
```bash
./run_agent_implementation.sh --phase 4
```

### Phase 5: Testing
**Agent**: Testing Agent
**Purpose**: Generate test plans and test skeletons
**Output**: `cache/phase_5.json`
**Actions**:
- Creates test plan matrix (1x1, 2x2, 2x4, 4x4 blocks)
- Generates test skeletons (optional)
- Documents testing approach

**Run Standalone**:
```bash
./run_agent_implementation.sh --phase 5
```

### Phase 6: Build & Verify
**Agent**: Verification Agent
**Purpose**: Build project and verify changes
**Output**: `cache/phase_6.json`
**Actions**:
- Runs `clang-format` syntax checks
- Validates linter compliance
- Runs build (optional, takes time)
- Reports any errors

**Run Standalone**:
```bash
./run_agent_implementation.sh --phase 6
```

### Phase 7: Final Review
**Agent**: Review Agent
**Purpose**: Generate summary and final approval
**Output**: `IMPLEMENTATION_SUMMARY.md`, `cache/phase_7.json`
**Actions**:
- Aggregates results from all phases
- Counts functions added
- Lists files modified
- Generates comprehensive summary

**Run Standalone**:
```bash
./run_agent_implementation.sh --phase 7
```

## API Configuration

The scripts use API configuration from `~/.bashrc`:

```bash
export ANTHROPIC_BASE_URL="https://litellm-proxy--tenstorrent.workload.tenstorrent.com"
export ANTHROPIC_MODEL="anthropic/claude-sonnet-4-20250514"
export ANTHROPIC_SMALL_FAST_MODEL="anthropic/claude-3-5-haiku-20241022"
export ANTHROPIC_CUSTOM_HEADERS="anthropic-beta: interleaved-thinking-2025-05-14"
export ANTHROPIC_API_KEY="sk-..."
```

**Supported Providers**:
- Anthropic Claude (recommended)
- OpenAI GPT-4

## Caching

Results from each phase are cached in `.cache/phase_N.json`. This allows:
- Resuming from any phase if earlier phase fails
- Skipping API calls on re-runs
- Debugging individual phases

**Clear Cache**:
```bash
rm -rf .cache/
```

## Prerequisites

### Required
- Python 3.8+
- Access to `/localdev/ncvetkovic/reconfig/tt-metal` repository
- Branch: `ncvetkovic/35739_add_missing_functions`
- API key: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

### Optional Python Packages
```bash
pip install anthropic  # For Anthropic Claude API
pip install openai     # For OpenAI GPT-4 API
```

**Note**: The script will work without these packages in `--skip-api` mode.

## Troubleshooting

### "No API key found"
```bash
# Set API key in bashrc or export it:
export ANTHROPIC_API_KEY="your-key-here"
source ~/.bashrc
```

### "Phase N failed"
```bash
# Check cache for error details:
cat .cache/phase_N.json

# Re-run just that phase:
./run_agent_implementation.sh --phase N --verbose
```

### "Repository not found"
```bash
# Verify path:
ls -la /localdev/ncvetkovic/reconfig/tt-metal

# Update REPO_PATH in add_block_variants.py if different
```

### "clang-format failed"
```bash
# Check syntax errors in modified files
clang-format --dry-run --Werror tt_metal/include/compute_kernel_api/eltwise_binary.h
```

## Dry Run Mode

Test the automation without making any changes:

```bash
./run_agent_implementation.sh --dry-run --verbose
```

This will:
- Show what would be done
- Generate templates in cache
- NOT modify any source files
- NOT run builds

## Skip API Mode

Run without making API calls (uses cached results):

```bash
./run_agent_implementation.sh --skip-api
```

Useful for:
- Testing script logic
- Re-running after fixes
- Avoiding API costs

## Output Files

### Generated by Automation
- `.cache/phase_*.json` - Cached results from each phase
- `BLOCK_VARIANTS_API.md` - API reference documentation
- `IMPLEMENTATION_SUMMARY.md` - Final summary report

### Modified by Automation
- `tt_metal/include/compute_kernel_api/eltwise_binary.h`
- `tt_metal/include/compute_kernel_api/reduce.h`
- `tt_metal/include/compute_kernel_api/pack.h`

## Manual Steps After Automation

1. **Review Generated Code**:
   ```bash
   cd /localdev/ncvetkovic/reconfig/tt-metal
   git diff tt_metal/include/compute_kernel_api/
   ```

2. **Build**:
   ```bash
   export TT_METAL_HOME=$(pwd)
   export PYTHONPATH=$(pwd)
   ./build_metal.sh
   ```

3. **Test** (if tests implemented):
   ```bash
   source python_env/bin/activate
   pytest tests/ -v
   ```

4. **Commit**:
   ```bash
   git add tt_metal/include/compute_kernel_api/
   git commit -m "#35739: Add block variants - automated implementation"
   ```

## Customization

### Add New Operations

Edit `phase1_inventory()` in `add_block_variants.py` to detect new operations:

```python
if 'div_tiles' in line:
    inventory["eltwise_binary"].append("div_tiles")
```

### Modify Template Pattern

Edit `generate_template()` in `add_block_variants.py` to change template structure.

### Change Model

Set different model in environment:
```bash
export ANTHROPIC_MODEL="anthropic/claude-opus-4-20250514"
```

## Support

For issues or questions:
1. Check logs in `.cache/`
2. Run with `--verbose` flag
3. Review `AGENT_PLAN_CONDENSED.md` for phase details
4. Check `TASK.md` for requirements

## License

Apache 2.0 - See LICENSE file in tt-metal repository
