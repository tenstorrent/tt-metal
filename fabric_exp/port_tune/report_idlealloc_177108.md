
## 2d [idlealloc] (10 rows in means; cycles/packet, pass a / pass b; base i4)
row                                                                i2               i4               i8
MEAN                                                  143.96/ 143.98   143.70/ 143.71   143.48/ 143.53 
MEAN - base                                            +0.27/  +0.26    +0.00/  +0.00    -0.21/  -0.19 
   i2: 2 rows moved beyond max(0.5, base spread) in both passes, same sign:
      Mesh3HopEW write 4352: +1.1 / +1.1
      Mesh3HopEW scatter_write 4352: +0.8 / +1.1
   i8: 1 rows moved beyond max(0.5, base spread) in both passes, same sign:
      Mesh3HopEW write 4352: -1.4 / -1.3

## vc1 [idlealloc] (13 rows in means; cycles/packet, pass a / pass b; base i4)
row                                                                i2               i4               i8
MEAN                                                  183.70/ 183.72   182.92/ 182.89   182.82/ 182.84 
MEAN - base                                            +0.78/  +0.83    +0.00/  +0.00    -0.09/  -0.06 
   i2: 7 rows moved beyond max(0.5, base spread) in both passes, same sign:
      VC1Mesh2HopMerge_size_16_ntype_unicast_write  : +2.4 / +2.4
      VC1Mesh2HopMerge_size_4352_ntype_unicast_write  : +2.0 / +1.9
      VC1BurstEW_N8_G500_size_16  : +1.4 / +1.2
      VC1BurstEW_N32_G500_size_16  : +1.0 / +1.3
      VC1Mesh1HopEW_size_16_ntype_unicast_write  : +1.0 / +1.3
      VC1MeshMulticast_size_4352_ntype_unicast_write  : +0.8 / +0.8
      VC1Mesh1HopNS_size_16_ntype_unicast_write  : +0.6 / +0.5

## vc1i [idlealloc] (8 rows in means; cycles/packet, pass a / pass b; base i4)
row                                                                i2               i4               i8
MEAN                                                  323.02/ 323.04   322.99/ 323.07   323.01/ 323.01 
MEAN - base                                            +0.03/  -0.03    +0.00/  +0.00    +0.02/  -0.06 
