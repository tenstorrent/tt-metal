# Yolov evaluation

- Using `coco-2017` validation dataset.
- Loading the dataset using `fiftyone` package.

### The below observations are for ttnn_model vs dataset(ground truth data):

The following model is evaluated(mAPval 50-95) for 500 samples.:-
-   YOLOv4(320x320 resolution) - **0.7562**
-   YOLOv4(640x640 resolution) - **0.7535**
-   YOLOv8s_World(640x640 resolution) - **0.7288**
-   YOLOv8x(640x640 resolution) - **0.7267**

Currently, The number of samples is set to 500.

To run the test of ttnn vs ground truth, please follow the following commands:

**YoloV4:** <br>
**_For 320x320,_**<br>
 ```sh
 pytest models/experimental/yolo_evaluation/yolo_common_evaluation.py::test_run_yolov4_eval[resolution0-1-act_dtype0-weight_dtype0-device_params0-tt_model]
 ```

**_For 640x640,_**<br>
 ```sh
 pytest models/experimental/yolo_evaluation/yolo_common_evaluation.py::test_run_yolov4_eval[resolution1-1-act_dtype0-weight_dtype0-device_params0-tt_model]
 ```

**YoloV8s_World:** <br>
**_For 640x640,_**<br>
 ```sh
 pytest models/experimental/yolo_evaluation/yolo_common_evaluation.py::test_yolov8s_world[res0-device_params0-tt_model]
 ```

**YoloV8x:** <br>
**_For 640x640,_**<br>
 ```sh
 pytest models/experimental/yolo_evaluation/yolo_common_evaluation.py::test_yolov8x[res0-device_params0-tt_model]
 ```

### The below observations are for torch_model vs dataset(ground truth data):

The following model is evaluated(mAPval 50-95) for 500 samples.:-
-   YOLOv4(320x320 resolution) - **0.7610**
-   YOLOv4(640x640 resolution) - **0.8029**
-   YOLOv8s_World(640x640 resolution) - **0.7102**
-   YOLOv8x(640x640 resolution) - **0.8116**

To run the test of ttnn vs ground truth, please follow the following commands:

**YOLOv4:** <br>
**_For 320x320,_**<br>
```sh
pytest models/experimental/yolo_evaluation/yolo_common_evaluation.py::test_run_yolov4_eval[resolution0-1-act_dtype0-weight_dtype0-device_params0-torch_model]
```
**_For 640x640,_**<br>
```sh
pytest models/experimental/yolo_evaluation/yolo_common_evaluation.py::test_run_yolov4_eval[resolution1-1-act_dtype0-weight_dtype0-device_params0-torch_model]
```
**YoloV8s_World:** <br>
**_For 640x640,_**<br>
 ```sh
 pytest models/experimental/yolo_evaluation/yolo_common_evaluation.py::test_yolov8s_world[res0-device_params0-torch_model]
 ```

 **YoloV8x:** <br>
**_For 640x640,_**<br>
 ```sh
 pytest models/experimental/yolo_evaluation/yolo_common_evaluation.py::test_yolov8x[res0-device_params0-torch_model]
 ```
