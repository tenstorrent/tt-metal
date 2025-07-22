# Ideal Performance of Data Movement Primitives
Below are the results from our directed ideal tests for each data movement primitive.

Units are **Bytes/cycle**.

"All" corresponds to 64 tensix cores on Wormhole B0 and 110 tensix cores on Blackhole P100

| Primitive                         | Wormhole B0   | Blackhole |
| --------------------------------- | --------------| --------- |
| DRAM Read                         | 22            | 33        |
| DRAM Write                        | 21            | 34        |
| One To One                        | 29            | 60        |
| One From One                      | 28            | 60        |
| One To All (Unicast)              | 31            | 62        |
| One To All (Multicast)            | 15            | 24        |
| One To All (Multicast + Linked)   | 22            | 41        |
| One From All                      | 30            | 60        |
