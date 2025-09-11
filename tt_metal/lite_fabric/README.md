# Lite Fabric

## What is Lite Fabric?

This is a reduced variant of the Fabric Erisc Datamover (EDM) ethernet kernel intended to service basic reads and writes to remote devices.

Lite Fabric (LF) is capable of being launched at the UMD level, therefore it can run without any devices open. This is required as to bootstrap Metal Firmware and EDM we need to be able to do reads and writes to remote devices.

LF only works on tunnel depths up to 1 at the moment.

## High Level Overview

### Compilation

LF is compiled & linked with static addresses. 24KB is reserved for LF.

It's possible to only build LF and not launch it using the APIs in `build.hpp`. `LinkFabricLite()` will output both the `.elf` file and also a `.bin` which can be loaded directly to the start address.

### Initialization for tunnel depth = 1

1. Each ethernet core on the MMIO capable device is assigned an ethernet channel. The host writes a configuration object (`lite_fabric::FabricLiteConfig`) which contains metadata such as the binary size, address, state, and if the current core is MMIO capable.
2. The host launches a LF kernel on each MMIO device ethernet core. The host polls each core for the `READY` status.
3. Each ethernet core is responsible for initialization their connected ethernet core, which is the ethernet core on the other device connected through ethernet.
   - Write it's own binary and config to the connected core over the ethernet channel.
   - Set the program counter and de-assert reset state to make the connected core begin executing `main()`.
   - MMIO and connected ethernet complete a 2 way handshake.
   - Status becomes `READY`.

### Sender and Receiver Channels

LF uses the same sender and receiver channel objects as EDM. There is one sender channel and one receiver channel on each ethernet core.

- MMIO Ethernet core
    - Sender channel are used for sending packets to the connected receiver channel. All packets from the host go into this channel.
    - Receiver channel is used for receiving data from remote reads.
- Connected Ethernet core
   - Sender channel is used to send data back to the MMIO ethernet core.
   - Receiver channel receives packets and data.

#### Servicing Channels

This function is called on the device to service the sender and receiver channels. It is available after the first time LF is initialized on the device. Kernels should periodically call this function to ensure read/write requests from the host are not stalled. If you don't call this then no read/write requests will be processed and the host will most likely be waiting forever.

> `lite_fabric.hpp`
> ```cpp
> """
> Service Lite Fabric channels. Only works when called from an ethernet core.
> """
> inline void service_lite_fabric_channels()
> ```

#### Host Interface

A host interface can be obtained from a helper method in the `FabricLiteMemoryMap`

> `lite_fabric.hpp`
>
> ```cpp
> // Returns a Host Interface for the tunnel starting at the MMIO core
> static auto make_host_interface(const tt_cxy_pair& mmio_core)
> ```


##### High Level API

> `lite_fabric.hpp`
> ```cpp
> template <uint32_t NUM_SENDER_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
> struct HostToFabricLiteInterface {
>
> // Write NoC address on the connected device
> void write_any_len(
>    void* mem_ptr,
>    size_t size,
>    uint64_t dst_noc_addr,
>    uint8_t noc_index = lite_fabric::edm_to_local_chip_noc);
>
> // Blocking read NoC address on the connected device
> void read_any_len(
>    void* mem_ptr,
>    size_t size,
>    uint64_t src_noc_addr,
>    uint8_t noc_index = lite_fabric::edm_to_local_chip_noc);
>
> // Write register on the connected device directly using Ethernet
> void write_reg(uint32_t reg_address, uint32_t reg_value);
>
> // Write and read a barrier to all cores to ensure previous writes have landed
> void barrier();
>
> };
> ```

### Synchronization

Implementation is in `lite_fabric_channels.hpp`

- Host to MMIO Ethernet core
   - Ring buffer read/write pointers are used to sync the host and LF.
   - Host needs to read pointers off the device to check if the requests have been processed.
   - `HostToFabricLiteInterface::DeviceToHost`. Set by the device. Read by the host.
   - `HostToFabricLiteInterface::HostToDevice`. Set by the host. Written by the device.
   - > The pointers on the device and host might not always be in sync at any given time, but when a request needs to be made they must sync (host will read values from device) to check if there's free space to place the packet.

- LF <-> LF
  - Stream registers are used to track when there is work. Each core uses at least 2 registers
    - `to_receiver_0_pkts_sent_id`
      - Sender channel increments this on the connected receiver channel to tell them there is data to process.
      - Connected receiver channel decrements once done.
    - `to_sender_0_pkts_completed_id`
      - Receiver channel increments this on the connected sender channel to tell them the data has been processed so free space as opened up.
