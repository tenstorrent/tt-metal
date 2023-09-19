Environment Variables
#########################

TT_METAL_SLOW_DISPATCH_MODE
****************************************
If set, this disables running with a command queue and uses host "slow" dispatch.

TT_METAL_SINGLE_CORE_MODE
*************************
If set, this will return a 1x1 core grid for requests of the device worker core grid,
so for ops, receiving a 1x1 core grid to parallelize on will make it run single core.
Note that some ops may not support a single core mode due to requiring multiple cores like for mcasting.
