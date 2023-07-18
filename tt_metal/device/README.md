# Extra Information for Device Folder
### Versim Build
TT_METAL_VERSIM_DISABLED=0 TT_METAL_VERSIM_ROOT=<Path to Versim libraries> make build
TT_METAL_VERSIM_DISABLED=0 TT_METAL_VERSIM_ROOT=<Path to Versim libraries> make tests

Every test should automatically target versim if it exists

### Versim Related ENV Var Control Flags
| Flag Name | Default Value | Description                   |
|  :---:    |  :---:        |  :---                         |
| TT_METAL_VERSIM_DISABLED | 1 | 0, Versim backend is used instead of silicon |
| TT_METAL_VERSIM_ROOT |  User Set | Required and points to root repo of versim libs |
| TT_METAL_VERSIM_FULL_DUMP | 0 | 1, Enables FPU waves to be dumped (only supported for ARCH_NAME=wormhole_b0) |
| TT_METAL_VERSIM_FORCE_FULL_SOC_DESC | 0 | 1, Forces versim to use full grid for soc descriptor instead of using the custom 1x1 versim yaml |
| TT_METAL_VERSIM_DUMP_CORES | string | When set, will cause versim to dump waveform for that core see tt_device.h::tt_device_params for spec on string |
