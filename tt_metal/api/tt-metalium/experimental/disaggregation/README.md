This directory contains utilities and datastructures to facilitate disaggregated inference.

A tensor chunk location table (`KvChunkAddressTable`) is included. The table is used as a "common language" interface so various components that are particants in KV Cache lifecycle management (in the future weights management) can understand where given pieces of data are located.

The table is intended to be used to have a representation for where data (KV chunks) in a tensor are located. The tensor may be distributed across multiple hosts and devices.

Users of this datastructure include:
1) Models: to specify their kv cache tensor layout
2) KV Cache Migration Service: to consume the table to understand how to migrate data (where to read from and send to, how to breakdown high level requests)
3) KV Cache Tiering Service: To consume and produce the mapping to understand how to move KV cache data throughout different levels and locations in a cache swarm
4) Weights Caching Service: To consume and produce the mapping to understand how to distribute weights data when loading/running models
