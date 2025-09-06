# TTNN Visualizer

## Installation
Please refer to ttnn-visualizer repository for installtion instuctions and getting started [here](https://github.com/tenstorrent/ttnn-visualizer). 
This report will provide you with some examples of using the tool for performance analysis and optimizations for vision models. 

## Yolov11
We will dive into using the visualizer to analyze the perforamce of [Yolov11n](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/yolov11) model in this section. 

## Instrctions for a swift setup
Ensure you have at least `python 3.10`. You may create a virtual enviroment with `python >= 3.10`. 
`pip install ttnn-visualizer` 

From your terminal execute: 
`ttnn-visualizer`

In order to use the visualizer, you need to generate 2 directories one for the momory visualizer and one for tracy tool for a comprehensive analysis. 
Before running your desired model test for analysis, Ensure to export the following enviroment variables repalcing the name of your report with an identifiable name of your choice.

```bash
export TTNN_CONFIG_OVERRIDES='{
    "enable_fast_runtime_mode": false,
    "enable_logging": true,
    "report_name": "NAME_OF_YOUR_CHOICE",
    "enable_graph_report": false,
    "enable_detailed_buffer_report": true,
    "enable_detailed_tensor_report": false,
    "enable_comparison_mode": false
}'
```

Next, make sure to build metal with the profiler enabled: 

```bash
./build_metal.sh -p
```
Next, you may execute your desired pytest as follows: 
```bash
python -m tracy -p -r -v -m pytest models/demos/yolov11/tests/pcc/test_ttnn_yolov11.py
```
You should now find two directories on your path similar to: 
```bash
generated/profiler/reports/2025_09_05_20_44_57/
```
```bash
generated/ttnn/reports/NAME_OF_YOUR_CHOICE
```
Next on the visualizer GUI upload the 2 directories similar to: 


