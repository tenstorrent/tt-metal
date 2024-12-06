# YoloV4 640x640 demo

## How to run yolov4

Use the following command to run the yolov4 demo, NOTE: the following demos are intented for visualization. It is not the performant implementation yet.
```
pytest models/demos/yolov4/demo/demo.py"
```

Once you run the command, The output file named `ttnn_prediction_demo.jpg` will be generated.

Currently, Trace implementation is blocked with ```Enqueue Read Buffer cannot be used with tracing``` error. [#17907](https://github.com/tenstorrent/tt-metal/issues/17907)
