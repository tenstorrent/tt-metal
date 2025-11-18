# NCurses Rewrite Plan for tt_smi_umd

## Summary
Rewriting tt_smi_umd to use ncurses library (like nvtop/htop) for proper TUI with:
- ✅ Zero flickering (double-buffering built-in)
- ✅ Proper terminal resize handling
- ✅ Clean keyboard input
- ✅ Professional appearance

## Changes Required

### 1. CMakeLists.txt ✅ DONE
- Added `find_package(Curses REQUIRED)`
- Linked `${CURSES_LIBRARIES}`
- Added include directories

### 2. Code Structure Changes

#### Remove:
- `TerminalController` class (lines ~350-380)
- Manual `termios` handling
- Manual ANSI escape codes
- `std::stringstream screen_buffer` approach

#### Replace with ncurses:

```cpp
// Initialization (in run() or constructor)
initscr();              // Initialize ncurses
cbreak();               // Raw input mode
noecho();               // Don't echo keypresses
nodelay(stdscr, TRUE);  // Non-blocking getch()
keypad(stdscr, TRUE);   // Enable arrow keys
curs_set(0);            // Hide cursor

// Rendering loopRewrite with ncurses (proper solution, like nvtop)
    clear();  // Clear screen (buffered)

    // Draw content at specific positions
    mvprintw(row, col, "Device %d", device_id);
    attron(COLOR_PAIR(1));  // Color
    mvprintw(row, col, "DRAM");
    attroff(COLOR_PAIR(1));

    refresh();  // Flush buffer (flicker-free!)

    // Non-blocking input
    int ch = getch();
    if (ch == '1') current_view_ = 1;
    if (ch == 'q') break;

    napms(refresh_ms);  // Sleep in milliseconds
}

// Cleanup
endwin();
```

#### Color Support:
```cpp
start_color();
init_pair(1, COLOR_GREEN, COLOR_BLACK);   // DRAM
init_pair(2, COLOR_CYAN, COLOR_BLACK);    // L1
init_pair(3, COLOR_YELLOW, COLOR_BLACK);  // Temperature
init_pair(4, COLOR_RED, COLOR_BLACK);     // Warnings
init_pair(5, COLOR_BLUE, COLOR_BLACK);    // Clock
```

### 3. Key API Replacements

| Old (ANSI codes) | New (ncurses) |
|------------------|---------------|
| `std::cout << "\033[2J"` | `clear()` |
| `std::cout << "\033[H"` | `move(0, 0)` |
| `std::cout << Color::GREEN` | `attron(COLOR_PAIR(1))` |
| `std::cout << "text"` | `printw("text")` or `mvprintw(row, col, "text")` |
| `std::cout << std::flush` | `refresh()` |
| Terminal raw mode setup | `cbreak(); noecho()` |
| `term_ctrl.check_keypress()` | `getch()` |

### 4. Benefits

1. **Zero Flickering**: ncurses uses double buffering internally
2. **SIGWINCH Handling**: ncurses handles resize automatically
3. **Cleaner Code**: No manual escape codes
4. **Professional**: Same approach as nvtop, htop, vim
5. **Portable**: Works across different terminals

### 5. Implementation Steps

1. ✅ Update CMakeLists.txt
2. ✅ Add ncurses include
3. TODO: Remove TerminalController class
4. TODO: Add ncurses initialization in `run()`
5. TODO: Replace all `std::cout` with ncurses calls in rendering sections
6. TODO: Replace keyboard handling with `getch()`
7. TODO: Add `endwin()` cleanup
8. TODO: Test and verify no flickering

## Next Steps

The file is quite large (~1400 lines). Options:
1. **Incremental**: Gradually replace sections with ncurses
2. **New File**: Create `tt_smi_ncurses.cpp` from scratch with best practices
3. **Hybrid**: Keep ANSI fallback for non-watch mode, use ncurses only in watch mode

**Recommendation**: Option 3 (Hybrid) - Use ncurses only in watch mode:
```cpp
if (watch_mode) {
    run_ncurses_mode(refresh_ms, use_sysfs);
} else {
    run_simple_mode();  // Current ANSI approach for one-shot
}
```

This keeps backward compatibility while adding professional TUI for watch mode.
