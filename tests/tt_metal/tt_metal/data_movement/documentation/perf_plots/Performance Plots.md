# Data Movement Performance Plots

This file contains visual representations of performance results from our various data movement primitive tests. A separate set of results is present for both architectures (i.e. Wormhole_B0 and Blackhole).

In all plots, "Sender" or "Writer" refers to outgoing data movement primitives where the master core(s) issue(s) a write transaction to subordinate cores or to DRAM. RISCV 0/NOC 0 is typically used for these transactions. On the other hand, "Receiver" or "Reader" refers to incoming data movement primitives where the master core(s) issue(s) a read transaction. RISCV 1/NOC 1 is typically used for these transactions.

For each test case, two plots are given to provide meaning from both the Hardware and the Software perspectives. The "Transaction Size vs Duration" plot depicts how long data movement takes to complete for a specific combination of test parameters (i.e. number of transactions and transaction sizes). This may be useful for kernel developers to gauge the latency of their OPs. The "Transaction Size vs Bandwidth" plot depicts how much the NOC capacity is saturated by data movement kernels using different combination of test parameters. This may be useful to gauge the effective performance of different data movement scenarios.

For steps to reproduce these plots, refer to our general [README](../../README.md).

For more information on each test, refer to the README under each test primitive directory, e.g. [README](../../dram_unary/README.md) for DRAM tests

## Wormhole_B0

### DRAM Interleaved Packet Sizes

![DRAM Interleaved Packet Sizes](./wormhole_b0/DRAM%20Interleaved%20Packet%20Sizes.png)

Dram read bandwidth saturates at about 37 B/cycle, according to HW experiments. DRAM write bandwidth should saturate at 64 B/cycle, instead of 35 B/c. There may be some configuration problem with the dram controller/phy or this may be the physical limit of the dram.

For more information on this primitive, refer to [README](../../dram_unary/README.md).

### One to One Packet Sizes

![One to One Packet Sizes](./wormhole_b0/One%20to%20One%20Packet%20Sizes.png)

Bandwidth in steady state, with > 2KB packet sizes, is close to theoretical max. Under 2KB, the bandwidth is limitted by either the RISC latency or by the NOC sending from L1 latency.

For more information on this primitive, refer to [README](../../one_to_one/README.md).

### Loopback Packet Sizes

![Loopback Packet Sizes](./wormhole_b0/Loopback%20Packet%20Sizes.png)

Loopback will have similar characteristics to the one to one test, however it uses two ports to send and receive data, as such it is more likely to cause contention.

For more information on this primitive, refer to [README](../../loopback/README.md).

### One from One Packet Sizes

![One from One Packet Sizes](./wormhole_b0/One%20from%20One%20Packet%20Sizes.png)

Bandwidth in steady state, with > 2KB packet sizes, is close to theoretical max. Under 2KB, the bandwidth is limitted by the RISC latency.

For more information on this primitive, refer to [README](../../one_from_one/README.md).

### One to All Packet Sizes
#### Unicast
##### 2x2

![One to All Unicast 2x2 Packet Sizes](./wormhole_b0/One%20to%20All%202x2%20Packet%20Sizes.png)

This test sends to a small grid. The bandwidth characteristics are similar to the one to one test. Note that it may appear that multicast has lower bandwidth, however multicast sends less data and has much lower latency, so it is prefered to use multicast.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

##### 4x4

![One to All Unicast 4x4 Packet Sizes](./wormhole_b0/One%20to%20All%204x4%20Packet%20Sizes.png)

This test sends to a medium grid. The bandwidth characteristics are similar to the one to one test. As the grid size increases, the number of transactions needed to saturate NOC decreases because the NOC needs to send num cores more packets. Note that it may appear that multicast has lower bandwidth, however multicast sends less data and has much lower latency, so it is prefered to use multicast.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

##### 10x10

![One to All Unicast 10x10 Packet Sizes](./wormhole_b0/One%20to%20All%2010x10%20Packet%20Sizes.png)

This test sends to a large grid. The bandwidth characteristics are similar to the one to one test. As the grid size increases, the number of transactions needed to saturate NOC decreases because the NOC needs to send num cores more packets. Note that it may appear that multicast has lower bandwidth, however multicast sends less data and has much lower latency, so it is prefered to use multicast.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

#### Multicast Unlinked
##### 2x2

![One to All Multicast Unlinked 2x2 Packet Sizes](./wormhole_b0/One%20to%20All%20Multicast%202x2%20Packet%20Sizes.png)

This test sends to a small grid using unlinked multicast. Bandwidth degrades due to path reserve being done after every transaction.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

##### 5x5

![One to All Multicast Unlinked 5x5 Packet Sizes](./wormhole_b0/One%20to%20All%20Multicast%205x5%20Packet%20Sizes.png)

This test sends to a medium grid using unlinked multicast. Bandwidth degrades due to path reserve being done after every transaction. As the grid size increases, the number of write acks increases which degrades bandwidth.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

##### 11x10

![One to All Multicast Unlinked 11x10 Packet Sizes](./wormhole_b0/One%20to%20All%20Multicast%2011x10%20Packet%20Sizes.png)

This test sends to a large grid using unlinked multicast. Bandwidth degrades due to path reserve being done after every transaction. As the grid size increases, the number of write acks increases which degrades bandwidth.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

#### Multicast Linked
##### 2x2

![One to All Multicast Linked 2x2 Packet Sizes](./wormhole_b0/One%20to%20All%20Multicast%20Linked%202x2%20Packet%20Sizes.png)

This test sends to a small grid using linked multicast. Linked causes path reserve to be done only once for all transactions, as such performance approaches theoretical.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

##### 5x5

![One to All Multicast Linked 5x5 Packet Sizes](./wormhole_b0/One%20to%20All%20Multicast%20Linked%205x5%20Packet%20Sizes.png)

This test sends to a medium grid using linked multicast. Linked causes path reserve to be done only once for all transactions, as such performance approaches theoretical. As the grid size increases, the number of write acks increases which degrades bandwidth. Posted multicasts do not have this issue, however it is not safe to use posted multicast due to a hardware bug.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

##### 11x10

![One to All Multicast Linked 11x10 Packet Sizes](./wormhole_b0/One%20to%20All%20Multicast%20Linked%2011x10%20Packet%20Sizes.png)

This test sends to a large grid using linked multicast. Linked causes path reserve to be done only once for all transactions, as such performance approaches theoretical. As the grid size increases, the number of write acks increases which degrades bandwidth. Posted multicasts do not have this issue, however it is not safe to use posted multicast due to a hardware bug.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

### One from All Packet Sizes

![One from All Packet Sizes](./wormhole_b0/One%20from%20All%20Packet%20Sizes.png)

At small packet sizes, the bandwidth is limited by the RISC latency. As the packet size increases, the bandwidth approaches 64 B/cycle. Similar to the one from one test.

For more information on this primitive, refer to [README](../../one_from_all/README.md).

### All to All Packet Sizes

<!-- ![All to All Packet Sizes](./wormhole_b0) -->

### All from All Packet Sizes

<!-- ![All from All Packet Sizes](./wormhole_b0) -->


## Blackhole

### DRAM Interleaved Packet Sizes

![DRAM Interleaved Packet Sizes](./blackhole/DRAM%20Interleaved%20Packet%20Sizes.png)

Dram read bandwidth saturates at about 37 B/cycle, according to HW experiments. DRAM write bandwidth should saturate at 64 B/cycle, instead of 35 B/c. There may be some configuration problem with the dram controller/phy or this may be the physical limit of the dram.

For more information on this primitive, refer to [README](../../dram_unary/README.md).

### One to One Packet Sizes

![One to One Packet Sizes](./blackhole/One%20to%20One%20Packet%20Sizes.png)

Bandwidth in steady state, with > 2KB packet sizes, is close to theoretical max. Under 2KB, the bandwidth is limitted by either the RISC latency or by the NOC sending from L1 latency.

For more information on this primitive, refer to [README](../../one_to_one/README.md).

### Loopback Packet Sizes

![Loopback Packet Sizes](./blackhole/Loopback%20Packet%20Sizes.png)

Loopback will have similar characteristics to the one to one test, however it uses two ports to send and receive data, as such it is more likely to cause contention.

For more information on this primitive, refer to [README](../../loopback/README.md).

### One from One Packet Sizes

![One from One Packet Sizes](./blackhole/One%20from%20One%20Packet%20Sizes.png)

Bandwidth in steady state, with > 2KB packet sizes, is close to theoretical max. Under 2KB, the bandwidth is limitted by the RISC latency.

For more information on this primitive, refer to [README](../../one_from_one/README.md).

### One to All Packet Sizes
#### Unicast
##### 2x2

![One to All Unicast 2x2 Packet Sizes](./blackhole/One%20to%20All%202x2%20Packet%20Sizes.png)

This test sends to a small grid. The bandwidth characteristics are similar to the one to one test. Note that it may appear that multicast has lower bandwidth, however multicast sends less data and has much lower latency, so it is prefered to use multicast.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

##### 4x4

![One to All Unicast 4x4 Packet Sizes](./blackhole/One%20to%20All%204x4%20Packet%20Sizes.png)

This test sends to a medium grid. The bandwidth characteristics are similar to the one to one test. As the grid size increases, the number of transactions needed to saturate NOC decreases because the NOC needs to send num cores more packets. Note that it may appear that multicast has lower bandwidth, however multicast sends less data and has much lower latency, so it is prefered to use multicast.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

##### 10x10

![One to All Unicast 10x10 Packet Sizes](./blackhole/One%20to%20All%2010x10%20Packet%20Sizes.png)

This test sends to a large grid. The bandwidth characteristics are similar to the one to one test. As the grid size increases, the number of transactions needed to saturate NOC decreases because the NOC needs to send num cores more packets. Note that it may appear that multicast has lower bandwidth, however multicast sends less data and has much lower latency, so it is prefered to use multicast.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

#### Multicast Unlinked
##### 2x2

![One to All Multicast Unlinked 2x2 Packet Sizes](./blackhole/One%20to%20All%20Multicast%202x2%20Packet%20Sizes.png)

This test sends to a small grid using unlinked multicast. Bandwidth degrades due to path reserve being done after every transaction.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

##### 5x5

![One to All Multicast Unlinked 5x5 Packet Sizes](./blackhole/One%20to%20All%20Multicast%205x5%20Packet%20Sizes.png)

This test sends to a medium grid using unlinked multicast. Bandwidth degrades due to path reserve being done after every transaction. As the grid size increases, the number of write acks increases which degrades bandwidth.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

##### 11x10

![One to All Multicast Unlinked 11x10 Packet Sizes](./blackhole/One%20to%20All%20Multicast%2011x10%20Packet%20Sizes.png)

This test sends to a large grid using unlinked multicast. Bandwidth degrades due to path reserve being done after every transaction. As the grid size increases, the number of write acks increases which degrades bandwidth.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

#### Multicast Linked
##### 2x2

![One to All Multicast Linked 2x2 Packet Sizes](./blackhole/One%20to%20All%20Multicast%20Linked%202x2%20Packet%20Sizes.png)

This test sends to a small grid using linked multicast. Linked causes path reserve to be done only once for all transactions, as such performance approaches theoretical.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

##### 5x5

![One to All Multicast Linked 5x5 Packet Sizes](./blackhole/One%20to%20All%20Multicast%20Linked%205x5%20Packet%20Sizes.png)

This test sends to a medium grid using linked multicast. Linked causes path reserve to be done only once for all transactions, as such performance approaches theoretical. As the grid size increases, the number of write acks increases which degrades bandwidth. Posted multicasts do not have this issue, however it is not safe to use posted multicast due to a hardware bug.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

##### 11x10

![One to All Multicast Linked 11x10 Packet Sizes](./blackhole/One%20to%20All%20Multicast%20Linked%2011x10%20Packet%20Sizes.png)

This test sends to a large grid using linked multicast. Linked causes path reserve to be done only once for all transactions, as such performance approaches theoretical. As the grid size increases, the number of write acks increases which degrades bandwidth. Posted multicasts do not have this issue, however it is not safe to use posted multicast due to a hardware bug.

For more information on this primitive, refer to [README](../../one_to_all/README.md).

### One from All Packet Sizes

![One from All Packet Sizes](./blackhole/One%20from%20All%20Packet%20Sizes.png)

At small packet sizes, the bandwidth is limited by the RISC latency. As the packet size increases, the bandwidth approaches 64 B/cycle. Similar to the one from one test.

For more information on this primitive, refer to [README](../../one_from_all/README.md).

### All to All Packet Sizes

<!-- ![All to All Packet Sizes](./blackhole) -->

### All from All Packet Sizes

<!-- ![All from All Packet Sizes](./blackhole) -->
