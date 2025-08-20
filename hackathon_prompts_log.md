# Hackathon Prompts Log - Pool3D Discussion

## Pool3D Implementation Analysis Session

### 1. Initial Question
**User**: "ok can you tell me what are differences in implementation of pool2d and pool3d?"

### 2. Dimensional Complexity Discussion
**User**: "do we need to have T last dim or C last dim?"

### 3. Halo Exchange Analysis
**User**: "ok, but how halo can work with pool3d?"
**User**: "how conv3d uses halo?"

### 4. Implementation Strategy
**User**: "i want dummiest approach, so it does not need to be efficient"
**User**: "ok what should be basic case? maybe 1 core interleaved?"

### 5. Memory Constraints
**User**: "note that out L1 memory is 1.4MB. how big is tensor that can be stored on one core?"
**User**: "ok but we also need to have output and other things on core. how hard is to do interleaved version on multiple cores? how it would be splitted if we cannot use halo?"
**User**: "but if i split it in half, i would need some data from other cores?"

### 6. Conv3D Analysis
**User**: "how conv3d handles this?"
**User**: "please explain how vol2col works? also, note that pool op does not need transformation like im2col in conv2d"
**User**: "ok, and how reader knows which data it should load?"
**User**: "so basically we don't do vol2col in pool3d?"

### 7. Implementation Details
**User**: "ok tell me step by step what should reader kernel do, but in the simple way"
**User**: "and in what format is data that compute kernel is going to consume?"
**User**: "ok so we can for start do compute on RISC instead using FPU/SFPU?"

### 8. Memory Layout Questions
**User**: "ok, if we have tiled input, how we can now which tiles (pages) reader is going to fetch?"
**User**: "and it is possible that we would not need all data from tile? how is that handled?"
**User**: "ok so easier way is to have input in RM layout. so it means one page is whole channel?"
**User**: "nice!! so reader kernel will basically read stick it needs?"

### 9. Multi-Core Distribution
**User**: "and if we have interleaved solution, that means we need somehow to specify ranges on each core?"
**User**: "what are reducers and workers for conv3d?"

### 10. Kernel Architecture
**User**: "so can you give me quick overview what each kernel needs to do for an easiest approach?"
**User**: "so cb is going to have just one kernel window at the time, or more?"
**User**: "and writer is going to write in DRAM?"
**User**: "and what is compute doing?"

### 11. Documentation Request
**User**: "ok can you again remember all my prompts in write them in .md file?"

## Key Conclusions Reached:

1. **Layout**: Pool3D should use ROW_MAJOR input layout like Conv3D
2. **No Vol2Col**: Pool3D doesn't need volume-to-column transformation
3. **Simple Architecture**: Reader → Compute (RISC-V) → Writer
4. **Stick-Based**: One page = all channels at spatial position (t,h,w)
5. **Multi-Core**: Spatial range assignment without halo exchange
6. **Memory**: ~500K element tensors fit comfortably in 1.4MB L1
7. **No Reducers**: Unlike Conv3D, Pool3D processes channels independently

## Next Steps Identified:
- Create Pool3D configuration structure
- Implement simple reader kernel (stick-based reads)
- Implement RISC-V compute kernel (max/avg operations)
- Implement writer kernel (stick-based writes)
- Add operation registration following Pool2D patterns
