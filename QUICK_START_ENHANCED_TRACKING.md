# Quick Start: Enhanced Memory Tracking

## 🎯 What Changed

Your allocation tracking system now captures:
- **Buffer Names** - Know what each allocation is for
- **Stack Traces** - See where allocations come from

## 🚀 Quick Start (3 Steps)

### 1. Restart the Server
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
pkill allocation_ser
./allocation_server_poc
```

### 2. Run Your Test
```bash
cd /home/tt-metal-apv
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"
```

### 3. Check the Output
The server terminal will show enhanced dumps with:
- 📝 Buffer names
- 📍 Stack traces
- All the previous info (size, refs, age, etc.)

## ✅ What Works Now

All components updated:
- ✅ Server (`allocation_server_poc`)
- ✅ Python client (`allocation_client.py`)
- ✅ Monitor client (`allocation_monitor_client`)
- ✅ Dump script (`dump_remaining_buffers.py`)
- ✅ Cleanup fixture (`conftest.py`)

## 📊 Example Output

**Before** (old format):
```
🔹 ID: 0x1b233ee0 | Size: 16128 KB | PID: 845661 | Refs: 1 | Age: 108s
```

**After** (new format):
```
🔹 ID: 0x1b233ee0 | Size: 16128 KB | PID: 845661 | Refs: 1 | Age: 108s | 📝 model_weights_layer_0
   └─ 📍 model.py:234:load_weights -> ttnn.py:567:to_device -> allocation_client.py:140:allocate
```

## 🔧 Troubleshooting

**Problem**: "Connection refused" when running test
- **Solution**: Make sure the server is running first

**Problem**: No buffer names shown
- **Solution**: That's normal! C++ allocations in tt-metal don't pass names yet. Only Python-tracked allocations will have names. This helps you identify which buffers are from the framework vs your code.

**Problem**: Stack traces too long
- **Solution**: They're truncated to 512 chars automatically, showing last 5 frames

## 📚 Next Steps

To get buffer names from C++ allocations, you'll need to:
1. Find allocation call sites in `device.cpp`
2. Add calls to send buffer info to the server
3. Include descriptive names based on buffer usage

For now, you'll see:
- ✅ **Python allocations**: Full names & stack traces
- ⏳ **C++ allocations**: Just buffer IDs (can identify by size/age/pattern)

The detailed output will help you identify **what remains allocated** after cleanup!
