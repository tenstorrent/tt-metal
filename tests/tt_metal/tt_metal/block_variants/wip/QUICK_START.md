# 🚀 Quick Start - Block Variants Automation

## One-Line Usage

```bash
cd /localdev/ncvetkovic/reconfig && ./run_agent_implementation.sh
```

## Common Commands

```bash
# Full run (all 7 phases)
./run_agent_implementation.sh

# Dry run (see what would happen, no changes)
./run_agent_implementation.sh --dry-run

# Run specific phase
./run_agent_implementation.sh --phase 1

# Verbose output
./run_agent_implementation.sh --verbose

# Help
./run_agent_implementation.sh --help
```

## Phase Cheat Sheet

| Phase | What It Does | Time | Output |
|-------|-------------|------|--------|
| 1 | Scan for tile ops, find missing blocks | 30s | `.cache/phase_1.json` |
| 2 | Generate C++ templates | 30s | `.cache/phase_2.json` |
| 3 | Insert into header files | 1m | Modified `*.h` files |
| 4 | Generate documentation | 30s | `BLOCK_VARIANTS_API.md` |
| 5 | Create test plans | 20s | Test skeletons |
| 6 | Build & verify | 30m | Build logs |
| 7 | Final summary | 10s | `IMPLEMENTATION_SUMMARY.md` |

**Total**: ~35 minutes (mostly phase 6 build time)

## Prerequisites Check

```bash
# Check API key
echo $ANTHROPIC_API_KEY

# Check Python
python3 --version  # Need 3.8+

# Check repository
ls /localdev/ncvetkovic/reconfig/tt-metal

# Check branch
cd /localdev/ncvetkovic/reconfig/tt-metal && git branch
```

## After Automation

```bash
# 1. Review changes
cd /localdev/ncvetkovic/reconfig/tt-metal
git diff tt_metal/include/compute_kernel_api/

# 2. Build (takes ~30 minutes)
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
./build_metal.sh

# 3. Test (if implemented)
source python_env/bin/activate
pytest tests/ -v

# 4. Commit
git add tt_metal/include/compute_kernel_api/
git commit -m "#35739: Add block variants (automated)"
```

## Files Created

```
reconfig/
├── run_agent_implementation.sh    ← Main script (bash wrapper)
├── add_block_variants.py          ← Core logic (Python)
├── AUTOMATION_README.md           ← Full documentation
├── AUTOMATION_SUMMARY.md          ← This summary
├── QUICK_START.md                 ← This file
└── .cache/                        ← Phase results
    ├── phase_1.json
    ├── phase_2.json
    └── ...
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No API key found" | `source ~/.bashrc` or `export ANTHROPIC_API_KEY="..."` |
| "Phase N failed" | Check `.cache/phase_N.json` for errors |
| "Repository not found" | Verify path: `/localdev/ncvetkovic/reconfig/tt-metal` |
| "Wrong branch" | `git checkout ncvetkovic/35739_add_missing_functions` |
| "clang-format failed" | Fix syntax errors in modified `.h` files |

## Documentation

- **Quick Start**: This file
- **Full Guide**: `AUTOMATION_README.md`
- **Summary**: `AUTOMATION_SUMMARY.md`
- **Agent Plan**: `AGENT_PLAN_CONDENSED.md`
- **Task Details**: `TASK.md`

## Support

```bash
# Run help
./run_agent_implementation.sh --help

# Check logs
cat .cache/phase_*.json

# Review plan
cat AGENT_PLAN_CONDENSED.md
```

---

**Ready to go!** Just run: `./run_agent_implementation.sh`
