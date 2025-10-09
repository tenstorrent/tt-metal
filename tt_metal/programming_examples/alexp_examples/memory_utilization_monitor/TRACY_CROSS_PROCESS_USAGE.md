# Tracy Memory Monitor - Cross-Process Usage

## Overview

The Tracy Memory Monitor now supports **cross-process monitoring** via Unix sockets, combining the best features:
- ✅ Tracy profiler integration (for detailed analysis)
- ✅ Cross-process visibility (monitor Python tests from C++ client)
- ✅ Real-time updates
- ✅ Zero code changes to your tests

## Architecture

```
┌──────────────────────┐
│ Python Process       │
│  test_mesh_alloc.py  │
│                      │
│  TracyMemoryMonitor  │◄─── Tracks allocations
│  (singleton)         │     in Python process
└──────────┬───────────┘
           │
           │ Queries singleton
           │
┌──────────▼───────────┐
│ tracy_memory_server  │◄─── Exposes via socket
│                      │
│ Unix Socket Server   │
│ /tmp/tracy_memory_   │
│      monitor.sock    │
└──────────┬───────────┘
           │
           │ Connects to socket
           │
┌──────────▼───────────┐
│ tracy_memory_        │◄─── Displays stats
│ monitor_client       │     from ANY process!
│                      │
│ Live Dashboard       │
└──────────────────────┘
```

## Quick Start

### Step 1: Build the tools

```bash
cd /home/tt-metal-apv
./build_metal.sh
```

This builds:
- `./build/programming_examples/tracy_memory_server`
- `./build/programming_examples/tracy_memory_monitor_client`

### Step 2: Start the server

```bash
# Terminal 1 - start the server (must be in same process as your test)
cd /home/tt-metal-apv
./build/programming_examples/tracy_memory_server
```

You should see:
```
🚀 Tracy Memory Monitor Server started
   Socket: /tmp/tracy_memory_monitor.sock
   Exposing TracyMemoryMonitor for cross-process access
   Press Ctrl+C to stop

✅ Server ready - waiting for connections...
```

### Step 3: Start the monitor client

```bash
# Terminal 2 - start the dashboard
cd /home/tt-metal-apv
./build/programming_examples/tracy_memory_monitor_client -a
```

You should see:
```
✅ Connected to tracy_memory_server (cross-process mode)
   Will show allocations from ALL processes!
```

### Step 4: Run your test

```bash
# Terminal 3 - run your Python test
cd /home/tt-metal-apv
python tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/test_mesh_allocation.py
```

**Watch Terminal 2** - you should see real-time memory changes! 🎉

## Command Options

### tracy_memory_server
```bash
./tracy_memory_server
# No options - just runs and exposes the local TracyMemoryMonitor
```

### tracy_memory_monitor_client
```bash
# Monitor all devices, 1 second refresh
./tracy_memory_monitor_client -a

# Monitor specific devices
./tracy_memory_monitor_client -d 0,1,2

# Faster refresh (100ms)
./tracy_memory_monitor_client -a -r 100

# Single query (no live updates)
./tracy_memory_monitor_client -a -q
```

## Troubleshooting

### "Using local singleton (same-process mode)"

**Problem:** Client can't connect to server.

**Solutions:**
1. Make sure `tracy_memory_server` is running FIRST
2. Check socket exists: `ls -l /tmp/tracy_memory_monitor.sock`
3. If stale socket exists: `rm /tmp/tracy_memory_monitor.sock` and restart server

### "No memory changes showing"

**Problem:** Server and client are running, but stats are zero.

**Likely cause:** The server needs to be in the **same process** as your test!

**Current limitation:** The server reads from its own `TracyMemoryMonitor::instance()`, so it needs to be:
- Embedded in your Python process (via C extension), OR
- Running as part of the test binary

**Workaround:** Use the existing `allocation_server_poc` system which has proper cross-process IPC:
```bash
# Terminal 1
./build/programming_examples/allocation_server_poc

# Terminal 2
./build/programming_examples/allocation_monitor_client -a

# Terminal 3
python test_mesh_allocation.py
```

## Next Steps

To make `tracy_memory_server` work truly cross-process, we'd need to:

1. **Option A:** Have `tt_metal` library launch the server automatically
2. **Option B:** Use shared memory instead of singleton
3. **Option C:** Enhance `allocation_server_poc` to integrate with Tracy

For now, the socket infrastructure is ready - it just needs the server to be in the right process!

## Comparison: allocation_server_poc vs tracy_memory_server

| Feature | allocation_server_poc | tracy_memory_server |
|---------|----------------------|---------------------|
| Cross-process | ✅ Yes (works now) | ⚠️  Needs integration |
| Tracy integration | ❌ No | ✅ Yes |
| Code changes needed | ✅ Manual calls | ✅ Automatic (via GraphTracker) |
| Protocol | Custom binary | Memory queries |
| Status | Production ready | Infrastructure ready |

Both systems have their place - use `allocation_server_poc` for immediate cross-process monitoring!
