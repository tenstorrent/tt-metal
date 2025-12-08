# Kernel Debugging Tips

## TT-TRIAGE

* still in development, so it might not work seamlessly
* useful for hang debug
* if your test hangs, kill the process and run `./tools/tt-triage.py` --verbosity=4 --dev=0 to get stack traces on all RISCs on all cores (script is inside tt-metal repo)


## DPRINT

https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/kernel_print.html

* useful for deterministic hang debug and pcc issues
* simple example:
    * include `debug/dprint.h` header in your kernel
    * print variable from a kernel: `DPRINT << “my variable is ” << my_variable << ENDL();`
    * add `TT_METAL_DPRINT_CORES="(0,0)"`  to your pytest command to output dprints from the first core


### Printing data from CBs

* useful for deterministic hang debug and pcc issues
* if debugging compute kernel, use print_full_tile  from `tt_metal/hw/inc/debug/dprint_pages.h `
* if debugging reader/writer kernel print_bf16_pages /print_f32_pages from `tt_metal/hw/inc/debug/dprint_pages.h`, or you can use the following function (DPRINT_DATA0 / DPRINT_DATA1 depends if you debug reader/writer kernel):
```cpp
inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_DATA0({ DPRINT << r << " " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL(); });
    }
    DPRINT << "++++++" << ENDL();
}
```
* when calling function, specify which cb and which tile from that cb

## Watcher

https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/watcher.html

* useful for hang debug
* simple example:
    * add `TT_METAL_WATCHER=1` to your pytest command
    * when the test hangs, kill the process and look at `generated/watcher/watcher.log` file to look at the kernels and waypoints


## General tips

* If you are experiencing issues with a test that runs multiple operations, it’s good to try and minimize the test as much as possible, ideally to a unit test with one op only.
* If the op is running on multiple devices - reduce it to as few devices as possible. Same goes for core grid.
* Try turning program cache off, if you are using it.
* Try setting input values to fixed numbers (all zeros or ones, for example).
