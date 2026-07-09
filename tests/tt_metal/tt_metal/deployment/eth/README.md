# Ethernet Deployment Test Suite
This test suite is meant for testing the functionality of the Ethernet subsystem.

## Test cases
* **TensixDeploymentEthernet00LinkUp**
  This test only sends one packet across the link to test whether the links
  have been established.

* **TensixDeploymentEthernet01Bandwidth**
  This test saturates the link by repeatedly sending data from L1 over the
  Ethernet link. The initial data is written by the host to the sender's L1,
  and the contents are compared after all transactions have finished. The host
  downloads the data from the receiver side and compares it.

* **TensixDeploymentEthernet02BandwidthBidir**
  Similar as above but full-duplex, the host initializes both L1s and then
  downloads and compares both L1s after the transactions have finished.

* **TensixDeploymentEthernet03DataIntegrityDram**
  This test streams ~4GB of data from DRAM over Ethernet links, and compares
  the results to the original data. The sender side and the receiver side DRAM
  banks are initialized, and after the data transfer is finished, the receiver
  side source bank and destination bank are compared and mismatches are
  reported.

* **TensixDeploymentEthernet04DataIntegrityDramBidir**
  Similar as above but full-duplex.

## Environment variables

- `ETH_TEST_TRANSFER_SIZE`: Controls the size of transfers initiated through metal

### Tests using DRAM:
- `ETH_TEST_START_ADDR`: Starting address of DRAM to copy over
- `ETH_TEST_END_ADDR`:   Ending address of DRAM to copy over

### Tests not using DRAM:
- `ETH_TEST_TRANSFER_COUNT`: Number of times to initiate the transfer
