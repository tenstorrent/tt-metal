# NCurses Testing Guide for tt_smi_umd

## Quick Test

```bash
cd /home/ttuser/aperezvicente/tt-metal-apv
./build/programming_examples/tt_smi_umd -w
```

## What to Test

### 1. Basic Launch ✅
**Expected**: Tool should launch cleanly with no flickering

**Test**:
```bash
./build/programming_examples/tt_smi_umd -w
```

**Success criteria**:
- Screen clears immediately
- Shows device information
- No visible flicker or screen tearing
- Cursor is hidden

### 2. View Switching
**Expected**: Pressing keys 1, 2, 3 switches views smoothly

**Test**:
```
Press '1' → Main view (device table + memory breakdown)
Press '2' → Charts view (real-time graphs)
Press '3' → Detailed telemetry view
```

**Success criteria**:
- View changes instantly
- No flicker during transition
- Content updates properly
- No artifacts or overlapping text

### 3. Terminal Resize 🔥 CRITICAL TEST
**Expected**: No hanging, no crashes, graceful resize

**Test**:
1. Launch `tt_smi_umd -w`
2. Resize terminal window (drag corner or maximize/restore)
3. Try multiple resizes quickly
4. Make window very small, then very large

**Success criteria**:
- ✅ Tool does NOT hang
- ✅ No crash
- ✅ Content adapts to new size
- ✅ Charts adjust width/height
- ✅ Can still switch views and quit

**Previous behavior**: HUNG after ~10 seconds or on resize
**Expected behavior**: Smooth resize handling like nvtop

### 4. Flickering Test 🔥 PRIMARY GOAL
**Expected**: Zero visible flickering

**Test**:
1. Launch in watch mode
2. Observe screen updates for 30 seconds
3. Try all 3 views
4. Look for any screen tearing or flicker

**Success criteria**:
- ✅ No visible flicker
- ✅ Smooth updates every 500ms
- ✅ Text appears solid, not "flashing"
- ✅ Numbers update cleanly
- ✅ Graphs animate smoothly

**Previous behavior**: Visible flicker on every update
**Expected behavior**: Solid, flicker-free like nvtop/htop

### 5. High Refresh Rate Test
**Expected**: No lag or "sparking" even at fast updates

**Test**:
```bash
# Test with 100ms refresh (10 updates/sec)
./build/programming_examples/tt_smi_umd -w -i 100
```

**Success criteria**:
- No visible artifacts ("sparking")
- Smooth continuous updates
- Terminal stays responsive
- Can still quit with 'q'

**Previous behavior**: "sparks" at high refresh rates
**Expected behavior**: Smooth even at 100ms updates

### 6. Keyboard Responsiveness
**Expected**: Keys work instantly

**Test**:
1. Launch tool
2. Rapidly press 1, 2, 3, 1, 2, 3
3. Press 'q' to quit

**Success criteria**:
- Views switch immediately
- No key lag
- 'q' exits cleanly
- Terminal restored properly

### 7. Exit and Cleanup
**Expected**: Terminal restored to normal state

**Test**:
1. Launch tool
2. Press 'q' to quit
3. Check terminal state

**Success criteria**:
- Cursor visible again
- Terminal in normal mode
- Can type commands normally
- No leftover artifacts on screen

### 8. Memory Tracking Integration
**Expected**: Works with allocation server

**Test**:
```bash
# Terminal 1: Start allocation server
cd /home/ttuser/aperezvicente/tt-metal-apv
./build/programming_examples/allocation_server_poc

# Terminal 2: Run tt_smi_umd
./build/programming_examples/tt_smi_umd -w
```

**Success criteria**:
- Memory stats displayed
- All 4 memory types shown (DRAM, L1, L1_SMALL, TRACE)
- Utilization bars render correctly
- No flicker in memory section

### 9. Telemetry Display
**Expected**: All telemetry fields visible and updating

**Test**:
1. Press '1' for main view
2. Check temperature, power, clock displayed
3. Press '3' for detailed telemetry
4. Verify all fields present

**Success criteria**:
- Temperature shows (not "N/A" if device available)
- Power/TDP shows
- AICLK, AXICLK, ARCCLK show
- Values update periodically
- No corruption or "65535" values

### 10. Multi-Device Support
**Expected**: All devices shown properly

**Test** (if you have multiple devices):
```bash
./build/programming_examples/tt_smi_umd -w
```

**Success criteria**:
- All devices listed
- Each device has separate row in table
- Charts show all devices (View 2)
- Can track memory for each device independently

## Comparison: Before vs After

| Aspect | Before (ANSI) | After (ncurses) |
|--------|---------------|-----------------|
| **Flickering** | ❌ Visible on every update | ✅ Zero flicker |
| **Resize** | ❌ Hangs after 10 sec | ✅ Smooth |
| **High refresh** | ❌ "Sparking" artifacts | ✅ Smooth |
| **Code** | Complex ANSI escapes | Clean ncurses API |
| **Terminal restore** | Manual | Automatic |

## Known Issues / Limitations

### Current Implementation Notes
1. **OutputWrapper approach**: Uses stream-like wrapper to route to ncurses or cout
2. **Color handling**: ANSI `Color::` constants still used but converted to ncurses colors
3. **Layout**: Still uses same rendering logic, just different output backend

### Potential Future Improvements
- [ ] Native ncurses windows/panels for each view
- [ ] More sophisticated color schemes
- [ ] Mouse support (ncurses supports this!)
- [ ] Window/panel borders like nvtop

## Debugging

### If tool doesn't start
```bash
# Check ncurses is installed
ldconfig -p | grep ncurses

# Try one-shot mode (bypasses ncurses)
./build/programming_examples/tt_smi_umd
```

### If screen is garbled
```bash
# Reset terminal
reset

# Or
tput reset
```

### If cursor stays hidden after crash
```bash
# Show cursor manually
tput cnorm

# Or
echo -e "\033[?25h"
```

### If resize still hangs
- Check `SIGWINCH` handler is working
- Verify `endwin()` + `refresh()` sequence
- May need to add `clear()` after resize

## Success Criteria Summary

✅ All tests must pass:
1. ✅ Zero flickering
2. ✅ No hang on resize
3. ✅ Smooth at high refresh rates
4. ✅ Keyboard responsive
5. ✅ Clean exit
6. ✅ All views working
7. ✅ Telemetry displaying
8. ✅ Works with allocation server

## Report Results

After testing, report:
- ✅ What works perfectly
- ⚠️  What has minor issues
- ❌ What is broken

This will help prioritize any remaining fixes!
