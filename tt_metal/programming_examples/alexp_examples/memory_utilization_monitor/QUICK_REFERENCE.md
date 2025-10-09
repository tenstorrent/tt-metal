# Memory Monitoring - Quick Reference Card

## 🚀 Quick Start (30 seconds)

```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./build_allocation_server.sh
./demo_allocation_server.sh
```

## 📋 Cheat Sheet

### Allocation Server (Cross-Process Monitoring)

```bash
# Build everything
./build_allocation_server.sh

# Terminal 1: Start server
./allocation_server_poc

# Terminal 2: Start monitor
./allocation_monitor_client -r 500      # Refresh every 500ms
./allocation_monitor_client -d 1        # Monitor device 1

# Terminal 3: Run demo clients
./allocation_client_demo                # C++ client
./allocation_client.py                  # Python client
```

### In-Process Monitoring

```bash
# Build
cd /home/tt-metal-apv
cmake --build build-cmake --target memory_monitor_test -j

# Run with integrated test
./build/programming_examples/memory_monitor_test -t -r 1000

# Run without test (requires TT device)
./build/programming_examples/memory_monitor_test -r 500
```

## 🎯 Command Line Options

### allocation_monitor_client

| Option | Description | Example |
|--------|-------------|---------|
| `-r <ms>` | Refresh interval in milliseconds | `-r 500` |
| `-d <id>` | Device ID to monitor | `-d 0` |
| `-h` | Show help | `-h` |

### memory_monitor_test

| Option | Description | Example |
|--------|-------------|---------|
| `-t` | Run with integrated test | `-t` |
| `-r <ms>` | Refresh interval in milliseconds | `-r 1000` |

## 📁 File Quick Reference

| File | Purpose | Build Needed |
|------|---------|--------------|
| `allocation_server_poc` | Server daemon | `./build_allocation_server.sh` |
| `allocation_client_demo` | C++ demo client | `./build_allocation_server.sh` |
| `allocation_monitor_client` | Real-time monitor | `./build_allocation_server.sh` |
| `allocation_client.py` | Python client | No (script) |
| `memory_monitor_test` | In-process monitor | `cmake --build` |

## 🔍 Debugging

### Server won't start

```bash
# Check if socket already exists
ls -l /tmp/tt_allocation_server.sock

# Remove old socket
rm /tmp/tt_allocation_server.sock

# Check if server is already running
ps aux | grep allocation_server

# Kill old server
killall allocation_server_poc
```

### Client can't connect

```bash
# 1. Check server is running
ps aux | grep allocation_server

# 2. Check socket exists
ls -l /tmp/tt_allocation_server.sock

# 3. Check permissions
# Socket should be readable/writable

# 4. Start server if needed
./allocation_server_poc &
```

### Monitor shows zeros

```bash
# Make sure clients are reporting allocations!
# In another terminal:
./allocation_client_demo

# Or:
./allocation_client.py
```

## 🧪 Testing Scenarios

### Test 1: Single Client

```bash
# Terminal 1
./allocation_server_poc

# Terminal 2
./allocation_monitor_client -r 500

# Terminal 3
./allocation_client_demo
```

Expected: Monitor shows memory going up, then down

### Test 2: Multiple Clients

```bash
# Terminal 1
./allocation_server_poc

# Terminal 2
./allocation_monitor_client -r 500

# Terminal 3
./allocation_client_demo &
./allocation_client_demo &
./allocation_client.py &
```

Expected: Monitor shows combined memory from all clients

### Test 3: In-Process Test

```bash
./build/programming_examples/memory_monitor_test -t -r 1000
```

Expected: See memory allocations and deallocations in real-time

## 📊 Sample Output

### Server Output
```
🚀 Allocation Server started
   Listening on: /tmp/tt_allocation_server.sock
✓ [PID 12345] Allocated 1048576 bytes of L1 on device 0
✗ [PID 12345] Freed buffer 1 (1048576 bytes)
📊 Current Statistics:
  Device 0: DRAM: 100MB, L1: 4MB
```

### Monitor Output
```
═══════════════════════════════════════════════════════════
  Cross-Process Memory Monitor (via Allocation Server)
═══════════════════════════════════════════════════════════
Device 0 Statistics:
  Active Buffers: 12
  DRAM:   100.00 MB /  12.00 GB  [██░░░░░] 0.8%
  L1:       4.00 MB /  75.00 MB  [█████░░] 5.3%
```

## 🛠️ Build Targets

### Allocation Server

```bash
# All POC tools (no TT-Metal dependency)
./build_allocation_server.sh

# Individual builds
g++ -o allocation_server_poc allocation_server_poc.cpp -std=c++17 -pthread -O2
g++ -o allocation_client_demo allocation_client_demo.cpp -std=c++17 -pthread -O2
g++ -o allocation_monitor_client allocation_monitor_client.cpp -std=c++17 -pthread -O2
```

### In-Process Monitors

```bash
# All monitors (requires TT-Metal)
cd /home/tt-metal-apv
cmake --build build-cmake -j

# Individual targets
cmake --build build-cmake --target memory_monitor -j
cmake --build build-cmake --target memory_monitor_test -j
cmake --build build-cmake --target memory_monitor_simple -j
cmake --build build-cmake --target memory_monitor_minimal -j
```

## 📚 Documentation Quick Links

| Topic | File |
|-------|------|
| Getting started | [ALLOCATION_SERVER_README.md](ALLOCATION_SERVER_README.md) |
| Complete design | [ALLOCATION_SERVER_DESIGN.md](ALLOCATION_SERVER_DESIGN.md) |
| Architecture diagrams | [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) |
| Complete summary | [SUMMARY.md](SUMMARY.md) |
| Build instructions | [BUILD_ALLOCATION_SERVER.md](BUILD_ALLOCATION_SERVER.md) |
| Memory architecture | [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md) |
| In-process monitors | [README.md](README.md) |

## 🎓 Common Use Cases

### Use Case 1: Development

```bash
# Use in-process monitor for quick testing
./build/programming_examples/memory_monitor_test -t -r 1000
```

### Use Case 2: Multi-Process Development

```bash
# Start server once
./allocation_server_poc &

# Start monitor in tmux/screen
./allocation_monitor_client -r 500

# Run your apps normally
python train.py  # If instrumented
./your_app       # If instrumented
```

### Use Case 3: Production Monitoring

```bash
# Start server as systemd service
sudo systemctl start tt-allocation-server

# Monitor from dashboard
http://localhost:8080/dashboard

# Query via API
curl http://localhost:8080/api/device/0/stats
```

## 🔐 Security Notes

```bash
# Set socket permissions
chmod 0660 /tmp/tt_allocation_server.sock

# Restrict to specific group
chown root:ttusers /tmp/tt_allocation_server.sock

# Only allow group members to connect
# (Users must be in ttusers group)
```

## ⚡ Performance Tips

### Server

- Default accepts 128 concurrent clients
- Can handle ~50K ops/sec
- Use batching for high-frequency allocations
- Consider rate limiting for production

### Monitor

- Lower refresh rate for less overhead: `-r 2000`
- Higher refresh rate for development: `-r 100`
- Monitor uses minimal CPU (mostly sleeping)

### Clients

- Reporting overhead: ~50-100 microseconds per allocation
- Non-blocking sends to avoid application delays
- Consider batching for very high allocation rates

## 🐛 Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| "Failed to connect" | Server not running | Start `./allocation_server_poc` |
| "Address already in use" | Old socket file | `rm /tmp/tt_allocation_server.sock` |
| Monitor shows all zeros | No clients reporting | Run `./allocation_client_demo` |
| Server crashes | Socket cleanup failed | `rm /tmp/tt_allocation_server.sock` |
| Permission denied | Socket permissions | Check socket ownership/permissions |

## 📞 Getting Help

1. Check this quick reference
2. Read [ALLOCATION_SERVER_README.md](ALLOCATION_SERVER_README.md)
3. Review [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)
4. Check server logs: `cat server.log`
5. Check monitor logs: `cat monitor.log`

## 🎯 Next Steps

1. ✅ Run the demo: `./demo_allocation_server.sh`
2. 📖 Read the design: [ALLOCATION_SERVER_DESIGN.md](ALLOCATION_SERVER_DESIGN.md)
3. 🔧 Integrate with your app
4. 🚀 Deploy to production

---

**Need more details?** See [SUMMARY.md](SUMMARY.md) for complete overview.

**Ready to start?**

```bash
./demo_allocation_server.sh
```

Happy monitoring! 📊✨
