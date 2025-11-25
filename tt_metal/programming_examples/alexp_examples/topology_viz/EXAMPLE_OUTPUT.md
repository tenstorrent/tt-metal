# Example Output

This file shows example output from the topology visualizer tool.

## Single N300 Board (2 chips)

```
╔═══════════════════════════════════════════════════════════╗
║  TT-Metal Physical Topology & Ethernet Link Visualizer  ║
╚═══════════════════════════════════════════════════════════╝

Initializing cluster...
Discovering topology...

╔══════════════════════════════════════════════════════════════╗
║              CHIP DETAILS                                    ║
╚══════════════════════════════════════════════════════════════╝

 Chip ID   Architecture     Board Type       Eth Channels
----------------------------------------------------------
       0    Wormhole B0           N300                  2
       1    Wormhole B0           N300                  2

╔══════════════════════════════════════════════════════════════╗
║              ETHERNET LINK TOPOLOGY MATRIX                   ║
╚══════════════════════════════════════════════════════════════╝

          0     1
    -------------
  0 |     -     2
  1 |     2     -

Legend: ■ All links UP  ■ Partial UP  ■ All links DOWN  . No connection
Numbers indicate count of ethernet links between chips

╔══════════════════════════════════════════════════════════════╗
║              DETAILED ETHERNET LINK STATUS                   ║
╚══════════════════════════════════════════════════════════════╝

Chip 0
  CH 8 -> Chip  1 CH 0  [UP]
  CH 9 -> Chip  1 CH 1  [UP]

Chip 1
  CH 0 -> Chip  0 CH 8  [UP]
  CH 1 -> Chip  0 CH 9  [UP]

╔══════════════════════════════════════════════════════════════╗
║              SUMMARY                                         ║
╚══════════════════════════════════════════════════════════════╝

Total Chips:         2
Total Ethernet Links: 4
Links UP:            4
Links DOWN:          0
Link Uptime:         100.0%

Topology visualization complete!
```

## 4-Chip Mesh Configuration

```
╔═══════════════════════════════════════════════════════════╗
║  TT-Metal Physical Topology & Ethernet Link Visualizer  ║
╚═══════════════════════════════════════════════════════════╝

Initializing cluster...
Discovering topology...

╔══════════════════════════════════════════════════════════════╗
║              CHIP DETAILS                                    ║
╚══════════════════════════════════════════════════════════════╝

 Chip ID   Architecture     Board Type       Eth Channels
----------------------------------------------------------
       0    Wormhole B0           N300                  4
       1    Wormhole B0           N300                  4
       2    Wormhole B0           N300                  4
       3    Wormhole B0           N300                  4

╔══════════════════════════════════════════════════════════════╗
║              ETHERNET LINK TOPOLOGY MATRIX                   ║
╚══════════════════════════════════════════════════════════════╝

          0     1     2     3
    -------------------------
  0 |     -     2     2     .
  1 |     2     -     .     2
  2 |     2     .     -     2
  3 |     .     2     2     -

Legend: ■ All links UP  ■ Partial UP  ■ All links DOWN  . No connection
Numbers indicate count of ethernet links between chips

╔══════════════════════════════════════════════════════════════╗
║              DETAILED ETHERNET LINK STATUS                   ║
╚══════════════════════════════════════════════════════════════╝

Chip 0
  CH 8 -> Chip  1 CH 0  [UP]
  CH 9 -> Chip  1 CH 1  [UP]
  CH10 -> Chip  2 CH 0  [UP]
  CH11 -> Chip  2 CH 1  [UP]

Chip 1
  CH 0 -> Chip  0 CH 8  [UP]
  CH 1 -> Chip  0 CH 9  [UP]
  CH 8 -> Chip  3 CH 0  [UP]
  CH 9 -> Chip  3 CH 1  [UP]

Chip 2
  CH 0 -> Chip  0 CH10  [UP]
  CH 1 -> Chip  0 CH11  [UP]
  CH 8 -> Chip  3 CH 8  [UP]
  CH 9 -> Chip  3 CH 9  [UP]

Chip 3
  CH 0 -> Chip  1 CH 8  [UP]
  CH 1 -> Chip  1 CH 9  [UP]
  CH 8 -> Chip  2 CH 8  [UP]
  CH 9 -> Chip  2 CH 9  [UP]

╔══════════════════════════════════════════════════════════════╗
║              SUMMARY                                         ║
╚══════════════════════════════════════════════════════════════╝

Total Chips:         4
Total Ethernet Links: 16
Links UP:            16
Links DOWN:          0
Link Uptime:         100.0%

Topology visualization complete!
```

## With Link Failures

```
╔══════════════════════════════════════════════════════════════╗
║              ETHERNET LINK TOPOLOGY MATRIX                   ║
╚══════════════════════════════════════════════════════════════╝

          0     1
    -------------
  0 |     -     2
  1 |     2     -

Legend: ■ All links UP  ■ Partial UP  ■ All links DOWN  . No connection
Numbers indicate count of ethernet links between chips

╔══════════════════════════════════════════════════════════════╗
║              DETAILED ETHERNET LINK STATUS                   ║
╚══════════════════════════════════════════════════════════════╝

Chip 0
  CH 8 -> Chip  1 CH 0  [UP]
  CH 9 -> Chip  1 CH 1  [DOWN]

Chip 1
  CH 0 -> Chip  0 CH 8  [UP]
  CH 1 -> Chip  0 CH 9  [DOWN]

╔══════════════════════════════════════════════════════════════╗
║              SUMMARY                                         ║
╚══════════════════════════════════════════════════════════════╝

Total Chips:         2
Total Ethernet Links: 4
Links UP:            2
Links DOWN:          2
Link Uptime:         50.0%
```

## Color Coding

- **Green**: Healthy links (all UP)
- **Yellow**: Degraded (some links UP, some DOWN)
- **Red**: Failed links (all DOWN)
- **Gray/Dim**: Inactive or no connection

## Use Cases

1. **Quick Health Check**: Instantly see if all chips are connected properly
2. **Troubleshooting**: Identify specific failed ethernet links
3. **Topology Verification**: Confirm cluster is wired as expected
4. **Documentation**: Generate topology diagrams for system records
5. **Monitoring**: Regular checks for link degradation
