# 📁 Files Overview

Complete directory structure and file purposes for the Block Variants automation project.

## 📂 Directory Structure

```
/localdev/ncvetkovic/reconfig/
│
├── 🤖 AUTOMATION SCRIPTS (NEW!)
│   ├── run_agent_implementation.sh    ⭐ Main entry point
│   ├── add_block_variants.py          ⚙️  Core implementation
│   ├── AUTOMATION_README.md           📖 Full guide
│   ├── AUTOMATION_SUMMARY.md          📊 Summary
│   ├── QUICK_START.md                 🚀 Quick reference
│   └── FILES_OVERVIEW.md              📁 This file
│
├── 📋 TASK DEFINITION
│   ├── TASK.md                        🎯 Main task description
│   ├── AGENT_PLAN_CONDENSED.md        📝 7-phase plan (short)
│   └── AGENT_PLAN.md                  📝 7-phase plan (detailed)
│
├── 📚 CONTEXT & REFERENCE
│   ├── CLAUDE.md                      🏗️  Repository infrastructure
│   ├── API_Abstraction_Layers.md      🔧 API layer explanation
│   └── Low Level Contract and API Split.txt  📐 Contracts & patterns
│
├── 📊 IMPLEMENTATION STATUS
│   └── IMPLEMENTATION_SUMMARY.md      ✅ What's been done
│
├── 🧪 TESTING
│   └── tt-metal/BLOCK_VARIANTS_TESTING.md  🔬 Test guide
│
├── 💾 CACHE (auto-generated)
│   └── .cache/
│       ├── phase_1.json
│       ├── phase_2.json
│       └── ... (phases 3-7)
│
└── 📦 REPOSITORY
    └── tt-metal/                      🏛️  Main codebase
        └── tt_metal/include/compute_kernel_api/
            ├── eltwise_binary.h       ✅ Modified
            ├── reduce_custom.h        ✅ Modified
            └── pack.h                 ✅ Modified
```

---

## 🤖 Automation Scripts

### **run_agent_implementation.sh** ⭐ **START HERE**
```bash
# What: Bash wrapper that runs the Python automation
# Why: User-friendly interface with environment setup
# Use: ./run_agent_implementation.sh [--phase N] [--dry-run] [--verbose]
```
**Key Features**:
- ✅ Environment validation
- ✅ Colored output
- ✅ Error handling
- ✅ Help system
- ✅ Sources ~/.bashrc for API keys

### **add_block_variants.py** ⚙️
```bash
# What: Core Python implementation of 7-phase plan
# Why: Contains all the logic for automation
# Use: python3 add_block_variants.py [options]
```
**Key Classes**:
- `AIAgent` - Handles LLM API calls
- `BlockVariantsImplementation` - Implements each phase
- Phase methods: `phase1_inventory()` through `phase7_review()`

### **AUTOMATION_README.md** 📖
```
# What: Complete documentation for automation scripts
# Why: Reference guide for all features and usage
# Read: Before running automation
```
**Sections**:
- Quick Start
- Phase descriptions
- API configuration
- Troubleshooting
- Examples

### **AUTOMATION_SUMMARY.md** 📊
```
# What: High-level overview and architecture
# Why: Understand what automation does
# Read: To get the big picture
```

### **QUICK_START.md** 🚀
```
# What: One-page quick reference
# Why: Fast access to common commands
# Read: For quick lookup
```

---

## 📋 Task Definition

### **TASK.md** 🎯 **CORE SPEC**
```
# What: Complete task specification
# Why: Defines what needs to be done
# Read: Required context for understanding the task
```
**Key Content**:
- Problem statement
- Compute API Contract
- Data flow patterns
- Implementation approach
- Code examples
- Checklist

### **AGENT_PLAN_CONDENSED.md** 📝
```
# What: 7-phase execution plan (short version)
# Why: Step-by-step implementation strategy
# Read: Before and during implementation
```
**Phases**:
1. Inventory
2. Template Generation
3. Code Integration
4. Documentation
5. Testing
6. Build & Verify
7. Final Review

### **AGENT_PLAN.md** 📝
```
# What: 7-phase plan with detailed examples
# Why: Extended guidance with code examples
# Read: When you need detailed context
```

---

## 📚 Context & Reference

### **CLAUDE.md** 🏗️
```
# What: Repository infrastructure guide
# Why: Explains how tt-metal is organized
# Read: For understanding repository structure
```
**Topics**:
- Directory structure
- Build system
- Testing infrastructure
- Hardware specifics
- Git workflow

### **API_Abstraction_Layers.md** 🔧
```
# What: Explanation of API layers
# Why: Understand where Compute API fits
# Read: For architectural context
```
**Layers** (bottom to top):
1. LLKs (Low Level Kernels)
2. Low Level API
3. **Compute API** ← Our focus
4. Compute Kernels

### **Low Level Contract and API Split.txt** 📐
```
# What: Contracts and patterns specification
# Why: Defines rules for API design
# Read: For understanding constraints
```
**Defines**:
- `*_dest` pattern (DEST→DEST)
- `*_block` pattern (L1→DEST)
- `pack_*_block` pattern (DEST→L1)
- `*_tensor` pattern (L1→L1)

---

## 📊 Implementation Status

### **IMPLEMENTATION_SUMMARY.md** ✅
```
# What: Summary of completed work
# Why: Track what's been implemented
# Read: To see current status
```
**Contains**:
- Functions implemented
- Files modified
- Key features
- Statistics

---

## 🧪 Testing

### **tt-metal/BLOCK_VARIANTS_TESTING.md** 🔬
```
# What: Testing guide for block variants
# Why: Instructions for validation
# Read: When ready to test
```
**Covers**:
- Test strategy
- Example tests
- How to run tests

---

## 💾 Cache Files

### **.cache/phase_N.json**
```json
# What: Results from each phase execution
# Why: Enable resumption and debugging
# Use: Automatic (created by scripts)
```
**Files**:
- `phase_1.json` - Inventory results
- `phase_2.json` - Generated templates
- `phase_3.json` - Integration log
- `phase_4.json` - Documentation metadata
- `phase_5.json` - Test plans
- `phase_6.json` - Build results
- `phase_7.json` - Final summary

---

## 📖 Reading Order

### For First-Time Users
1. **QUICK_START.md** - Get up to speed fast
2. **TASK.md** - Understand the problem
3. **AUTOMATION_README.md** - Learn how to use automation
4. Run: `./run_agent_implementation.sh --dry-run`

### For Developers
1. **CLAUDE.md** - Repository context
2. **API_Abstraction_Layers.md** - Architecture
3. **TASK.md** - Requirements
4. **Low Level Contract and API Split.txt** - Design patterns
5. **AGENT_PLAN_CONDENSED.md** - Implementation strategy
6. **add_block_variants.py** - Code review

### For AI Agents
1. **CLAUDE.md** - Infrastructure
2. **TASK.md** - Specification
3. **AGENT_PLAN_CONDENSED.md** - Execution plan
4. Execute phases sequentially

---

## 🎯 File Purpose Summary

| File | Purpose | Read When |
|------|---------|-----------|
| **run_agent_implementation.sh** | Run automation | Ready to implement |
| **add_block_variants.py** | Core logic | Debugging or customizing |
| **QUICK_START.md** | Quick commands | Need fast reference |
| **AUTOMATION_README.md** | Full guide | Learning how to use |
| **AUTOMATION_SUMMARY.md** | Overview | Understanding architecture |
| **TASK.md** | Requirements | Understanding problem |
| **AGENT_PLAN_CONDENSED.md** | Execution plan | Implementing manually |
| **CLAUDE.md** | Repository info | Learning codebase |
| **API_Abstraction_Layers.md** | Architecture | Understanding layers |
| **IMPLEMENTATION_SUMMARY.md** | Status | Checking progress |

---

## 🚀 What To Do Next

### Option 1: Use Automation (Recommended)
```bash
cd /localdev/ncvetkovic/reconfig
./run_agent_implementation.sh --dry-run  # Preview
./run_agent_implementation.sh            # Run for real
```

### Option 2: Manual Implementation
```bash
# Follow the agent plan
cat AGENT_PLAN_CONDENSED.md

# Implement phase by phase
# Use the plan as a checklist
```

### Option 3: Understand First
```bash
# Read in order:
cat QUICK_START.md
cat TASK.md
cat AUTOMATION_README.md
```

---

## 📞 Help & Support

**Quick Help**: `./run_agent_implementation.sh --help`

**Documentation**:
- Quick: `QUICK_START.md`
- Detailed: `AUTOMATION_README.md`
- Architecture: `AUTOMATION_SUMMARY.md`

**Debugging**:
- Check: `.cache/phase_*.json`
- Verbose: `--verbose` flag
- Dry run: `--dry-run` flag

**Context**:
- Task: `TASK.md`
- Plan: `AGENT_PLAN_CONDENSED.md`
- Status: `IMPLEMENTATION_SUMMARY.md`

---

**Last Updated**: 2026-01-16
**Issue**: [#35739](https://github.com/tenstorrent/tt-metal/issues/35739)
**Status**: ✅ Ready to use
