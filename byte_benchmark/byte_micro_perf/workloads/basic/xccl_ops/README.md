## ccl_ops

### [all_reduce](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)
```python
torch.distributed.all_reduce(tensor, op=<RedOpType.SUM: 0>, group=None, async_op=False)
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size, dim_size] |



### [reduce_scatter](https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter_tensor)
```python
torch.distributed.reduce_scatter_tensor(output, input, op=<RedOpType.SUM: 0>, group=None, async_op=False)[source]
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size // world_size, dim_size] |



### [all_gather](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_into_tensor)
```python
torch.distributed.all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False)
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size // world_size, dim_size] |
| output_tensor | [batch_size, dim_size] |


### [all_to_all](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single)
```python
torch.distributed.all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False)[source]
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size, dim_size] |



### [broadcast](https://pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast)
```python
torch.distributed.broadcast(tensor, src=None, group=None, async_op=False, group_src=None)
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size, dim_size] |

Iterate on world_size, if world_size == 2,
1. broadcast from src 0 to other ranks
2. broadcast from src 1 to other ranks



### [p2p](https://pytorch.org/docs/stable/distributed.html#torch.distributed.isend)
```python
torch.distributed.isend(tensor, dst, tag=0, group=None, async_op=False)
torch.distributed.irecv(tensor, src, tag=0, group=None, async_op=False)
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size, dim_size] |

| world_size | pattern |
| ----------- | ------------ |
| 2 | 0 --> 1 |
| 4 | 0 --> 1 --> 2 --> 3 |
| 8 | 0 --> 1 --> 2 --> 3 --> 4 --> 5 --> 6 --> 7 |
