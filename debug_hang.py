#!/usr/bin/env python3
import gdb
import time

class WatchLoop(gdb.Command):
    """Watch the hanging loop and print values"""
    
    def __init__(self):
        super(WatchLoop, self).__init__("watch_loop", gdb.COMMAND_USER)
    
    def invoke(self, arg, from_tty):
        # Attach to the process
        pid = arg if arg else "709313"
        
        # Set breakpoint on the function we're interested in
        try:
            # Breakpoint on is_non_mmio_cmd_q_full check
            bp = gdb.Breakpoint("*tt::umd::RemoteCommunicationLegacyFirmware::write_to_non_mmio")
            print(f"Set breakpoint at write_to_non_mmio")
        except:
            print("Could not set breakpoint")
            
        print("Continuing...")
        gdb.execute("continue", to_string=True)
        
        # Try to examine registers when we hit
        for i in range(10):
            try:
                frame = gdb.selected_frame()
                print(f"\n=== Iteration {i} ===")
                print(f"Frame: {frame.name()}")
                
                # Try to print registers
                rdi = gdb.parse_and_eval("$rdi")
                rsi = gdb.parse_and_eval("$rsi")
                rdx = gdb.parse_and_eval("$rdx")
                rcx = gdb.parse_and_eval("$rcx")
                
                print(f"RDI: {rdi:#x}")
                print(f"RSI: {rsi:#x}")
                print(f"RDX: {rdx:#x}")
                print(f"RCX: {rcx:#x}")
                
                # Try to examine stack
                rsp = gdb.parse_and_eval("$rsp")
                print(f"Stack values near RSP:")
                for j in range(20):
                    addr = int(rsp) + j * 4
                    val = gdb.parse_and_eval(f"*(uint32_t*){addr:#x}")
                    print(f"  [{addr:#x}] = {val:#010x}")
                    
                gdb.execute("continue", to_string=True)
            except Exception as e:
                print(f"Error: {e}")
                break

WatchLoop()

