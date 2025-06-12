# Ideal Performance of Data Movement Primitives
Below are the results from our directed ideal tests for each data movement primitive.

Units are **Bytes/cycle**.

| Primitive                         | Wormhole B0   | Blackhole |
| --------------------------------- | --------------| --------- |
| DRAM Read                         | 22            | 33        |
| DRAM Write                        | 21            | 34        |
| One To One                        | 29            | 60        |
| One From One                      | 28            | 60        |
| One To All (Multicast + Linked)   | 19            | 34        |
| One From All                      | 30            | 60        |
