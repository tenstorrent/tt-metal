# Tools
General folder for various debug and validation binaries, libraries, and scripts

## Scripts
So far, scripts include

<ol>
    <li>memset.py, a script that allows you to write a vector to either L1 or DRAM given a chip id, start address, a size, and a value you want to write. If writing to DRAM, it goes through all channels and subchannels and writes the vector. If writing to L1, it goes through all cores and writes the vector. Example usage: </li>

    python3 tt_metal/tools/memset.py --mem_type dram --chip_id 0 --start_addr 0 --size 4 --val 0
</ol>

## Libraries

The `Profiler` is a debug library to be used to profile functions inside this repo. Refer to the
readme inside the folder for more info.
