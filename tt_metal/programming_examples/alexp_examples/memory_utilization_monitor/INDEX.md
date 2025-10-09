# Memory Utilization Monitor - Documentation Index

## 📖 Start Here

**New to this project? Start here:**

1. 🎯 **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Get started in 30 seconds
2. 📊 **[ALLOCATION_SERVER_README.md](ALLOCATION_SERVER_README.md)** - Overview and quick start
3. 🏗️ **[ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)** - Visual guide to the system

**Want the complete picture?**

4. 📋 **[SUMMARY.md](SUMMARY.md)** - Comprehensive overview of all tools

---

## 📚 Documentation by Category

### 🚀 Getting Started

| Document | Description | Time |
|----------|-------------|------|
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Cheat sheet and quick commands | 2 min |
| [ALLOCATION_SERVER_README.md](ALLOCATION_SERVER_README.md) | Quick start guide for allocation server | 5 min |
| [README.md](README.md) | Guide to in-process monitors | 5 min |
| [BUILD_ALLOCATION_SERVER.md](BUILD_ALLOCATION_SERVER.md) | Detailed build and run instructions | 10 min |

### 🏗️ Architecture & Design

| Document | Description | Time |
|----------|-------------|------|
| [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) | Visual architecture guide | 10 min |
| [ALLOCATION_SERVER_DESIGN.md](ALLOCATION_SERVER_DESIGN.md) | Complete design specification | 30 min |
| [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md) | TT-Metal memory system overview | 15 min |
| [RUNTIME_TRACKING.md](RUNTIME_TRACKING.md) | How runtime tracking works | 10 min |

### 🔍 Reference & Details

| Document | Description | Time |
|----------|-------------|------|
| [SUMMARY.md](SUMMARY.md) | Complete project summary | 15 min |
| [LIMITATIONS.md](LIMITATIONS.md) | Known limitations | 5 min |

### 🛠️ Build & Development

| Document | Description | Time |
|----------|-------------|------|
| [BUILD_ALLOCATION_SERVER.md](BUILD_ALLOCATION_SERVER.md) | How to build everything | 10 min |
| [CMakeLists.txt](CMakeLists.txt) | Build configuration | - |
| [build_allocation_server.sh](build_allocation_server.sh) | Automated build script | - |
| [demo_allocation_server.sh](demo_allocation_server.sh) | Automated demo script | - |

---

## 🎯 Documentation by Use Case

### "I want to monitor memory in my single-process application"

1. Read [README.md](README.md) - In-process monitors overview
2. Read [RUNTIME_TRACKING.md](RUNTIME_TRACKING.md) - How it works
3. Build and run `memory_monitor_with_test.cpp`

**Relevant files:**
- `memory_monitor.cpp`
- `memory_monitor_with_test.cpp`

---

### "I need to monitor memory across multiple processes"

1. Read [ALLOCATION_SERVER_README.md](ALLOCATION_SERVER_README.md) - Quick start
2. Read [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) - Visual guide
3. Run `./demo_allocation_server.sh`
4. Read [ALLOCATION_SERVER_DESIGN.md](ALLOCATION_SERVER_DESIGN.md) - Full design

**Relevant files:**
- `allocation_server_poc.cpp`
- `allocation_client_demo.cpp`
- `allocation_monitor_client.cpp`
- `allocation_client.py`

---

### "I want to understand where memory info comes from"

1. Read [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md) - Complete overview
2. Read [RUNTIME_TRACKING.md](RUNTIME_TRACKING.md) - How tracking works
3. Read [LIMITATIONS.md](LIMITATIONS.md) - What's possible and what's not

---

### "I want to integrate this into production"

1. Read [ALLOCATION_SERVER_DESIGN.md](ALLOCATION_SERVER_DESIGN.md) - Full design
2. Read [BUILD_ALLOCATION_SERVER.md](BUILD_ALLOCATION_SERVER.md) - Deployment
3. Read [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) - Deployment topology
4. Review [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Security & performance

---

### "I want to contribute or extend this"

1. Read [SUMMARY.md](SUMMARY.md) - Complete overview
2. Read [ALLOCATION_SERVER_DESIGN.md](ALLOCATION_SERVER_DESIGN.md) - Design details
3. Study the code:
   - `allocation_server_poc.cpp` - Server implementation
   - `allocation_client_demo.cpp` - Client example
   - `allocation_monitor_client.cpp` - Monitor example
4. Check [ALLOCATION_SERVER_DESIGN.md](ALLOCATION_SERVER_DESIGN.md) - Future work section

---

## 📂 Files by Type

### 📄 Documentation (Markdown)

| File | Category | Audience |
|------|----------|----------|
| [INDEX.md](INDEX.md) | Navigation | All |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Quick start | Users |
| [README.md](README.md) | Getting started | Users |
| [ALLOCATION_SERVER_README.md](ALLOCATION_SERVER_README.md) | Getting started | Users |
| [SUMMARY.md](SUMMARY.md) | Overview | All |
| [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) | Architecture | Developers |
| [ALLOCATION_SERVER_DESIGN.md](ALLOCATION_SERVER_DESIGN.md) | Design | Architects |
| [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md) | Technical | Developers |
| [RUNTIME_TRACKING.md](RUNTIME_TRACKING.md) | Technical | Developers |
| [LIMITATIONS.md](LIMITATIONS.md) | Reference | Developers |
| [BUILD_ALLOCATION_SERVER.md](BUILD_ALLOCATION_SERVER.md) | Build | DevOps |

### 💻 Source Code (C++)

#### In-Process Monitors (Require TT-Metal)

| File | Purpose | Build Target |
|------|---------|--------------|
| `memory_monitor.cpp` | Main in-process monitor | `memory_monitor` |
| `memory_monitor_with_test.cpp` | Monitor with integrated test | `memory_monitor_test` |
| `memory_monitor_simple.cpp` | Simplified monitor | `memory_monitor_simple` |
| `memory_monitor_minimal.cpp` | Minimal monitor | `memory_monitor_minimal` |

#### Cross-Process Tracking (Standalone)

| File | Purpose | Build |
|------|---------|-------|
| `allocation_server_poc.cpp` | Central tracking server | `./build_allocation_server.sh` |
| `allocation_client_demo.cpp` | Demo client | `./build_allocation_server.sh` |
| `allocation_monitor_client.cpp` | Monitor client | `./build_allocation_server.sh` |

### 🐍 Python Scripts

| File | Purpose |
|------|---------|
| `allocation_client.py` | Python client for allocation server |
| `test_runtime_tracking.py` | Test script (limited - separate process) |
| `test_persistent_memory.py` | Persistent allocation test |

### 🔧 Build & Automation

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | CMake build configuration |
| `build_allocation_server.sh` | Build script for allocation server |
| `demo_allocation_server.sh` | Automated demo |

---

## 🎓 Learning Paths

### Path 1: Beginner (15 minutes)

```
1. QUICK_REFERENCE.md          [2 min]
   ↓
2. Run demo_allocation_server.sh [5 min]
   ↓
3. ALLOCATION_SERVER_README.md  [5 min]
   ↓
4. ARCHITECTURE_DIAGRAM.md      [3 min]
```

**You'll learn:** What the system does and how to run it

---

### Path 2: User (30 minutes)

```
1. QUICK_REFERENCE.md              [2 min]
   ↓
2. ALLOCATION_SERVER_README.md     [5 min]
   ↓
3. BUILD_ALLOCATION_SERVER.md      [10 min]
   ↓
4. ARCHITECTURE_DIAGRAM.md         [10 min]
   ↓
5. Run and experiment                [3 min]
```

**You'll learn:** How to use and deploy the tools

---

### Path 3: Developer (1 hour)

```
1. SUMMARY.md                      [15 min]
   ↓
2. MEMORY_ARCHITECTURE.md          [15 min]
   ↓
3. ARCHITECTURE_DIAGRAM.md         [10 min]
   ↓
4. RUNTIME_TRACKING.md             [10 min]
   ↓
5. Study source code               [10 min]
```

**You'll learn:** How everything works internally

---

### Path 4: Architect (2 hours)

```
1. SUMMARY.md                         [15 min]
   ↓
2. ALLOCATION_SERVER_DESIGN.md        [30 min]
   ↓
3. ARCHITECTURE_DIAGRAM.md            [10 min]
   ↓
4. MEMORY_ARCHITECTURE.md             [15 min]
   ↓
5. All source code                    [30 min]
   ↓
6. BUILD_ALLOCATION_SERVER.md         [10 min]
   ↓
7. LIMITATIONS.md + Future Work       [10 min]
```

**You'll learn:** Complete system design and implementation

---

## 🎯 Quick Access

### Common Questions

| Question | Answer |
|----------|--------|
| How do I get started? | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |
| What's the architecture? | [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) |
| How do I build? | [BUILD_ALLOCATION_SERVER.md](BUILD_ALLOCATION_SERVER.md) |
| How does it work? | [ALLOCATION_SERVER_DESIGN.md](ALLOCATION_SERVER_DESIGN.md) |
| Where does memory info come from? | [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md) |
| What are the limitations? | [LIMITATIONS.md](LIMITATIONS.md) |
| How do I deploy to production? | [BUILD_ALLOCATION_SERVER.md](BUILD_ALLOCATION_SERVER.md) |

### Quick Commands

```bash
# Build everything
./build_allocation_server.sh

# Run demo
./demo_allocation_server.sh

# Manual 3-terminal setup
./allocation_server_poc              # Terminal 1
./allocation_monitor_client -r 500   # Terminal 2
./allocation_client_demo             # Terminal 3

# In-process monitor
cmake --build build-cmake --target memory_monitor_test -j
./build/programming_examples/memory_monitor_test -t -r 1000
```

---

## 📊 Documentation Statistics

| Category | Files | Total Words |
|----------|-------|-------------|
| Getting Started | 4 | ~8,000 |
| Architecture | 4 | ~15,000 |
| Reference | 2 | ~5,000 |
| Build | 3 | ~3,000 |
| **Total** | **13** | **~31,000** |

---

## 🗺️ Visual Navigation

```
INDEX.md (You are here!)
├── Quick Start
│   ├── QUICK_REFERENCE.md ⭐
│   ├── ALLOCATION_SERVER_README.md ⭐
│   └── README.md
│
├── Architecture
│   ├── ARCHITECTURE_DIAGRAM.md ⭐⭐⭐
│   ├── ALLOCATION_SERVER_DESIGN.md ⭐⭐
│   ├── MEMORY_ARCHITECTURE.md ⭐⭐
│   └── RUNTIME_TRACKING.md ⭐
│
├── Reference
│   ├── SUMMARY.md ⭐⭐
│   └── LIMITATIONS.md ⭐
│
└── Build
    ├── BUILD_ALLOCATION_SERVER.md ⭐
    └── CMakeLists.txt

⭐ = Essential reading
⭐⭐ = Recommended reading
⭐⭐⭐ = Must read!
```

---

## 🚀 Get Started Now!

**30-second quick start:**

```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./build_allocation_server.sh && ./demo_allocation_server.sh
```

**Then read:**
- [ALLOCATION_SERVER_README.md](ALLOCATION_SERVER_README.md) for overview
- [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) for visuals
- [ALLOCATION_SERVER_DESIGN.md](ALLOCATION_SERVER_DESIGN.md) for deep dive

---

## 📞 Support

Having trouble? Check:
1. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Troubleshooting section
2. [BUILD_ALLOCATION_SERVER.md](BUILD_ALLOCATION_SERVER.md) - Build issues
3. Server logs: `server.log`, `monitor.log`

---

**Happy monitoring!** 📊✨

*Last updated: October 6, 2025*
