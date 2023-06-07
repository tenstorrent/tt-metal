from pathlib import Path
import cv2
import torch
import math
import os
import glob
import numpy as np

import tt_lib


def torch2tt_tensor(py_tensor: torch.Tensor, tt_device):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    if size[-1] % 2 != 0:
        tt_device = tt_lib.device.GetHost()

    tt_tensor = tt_lib.tensor.Tensor(
        py_tensor.reshape(-1).tolist(),
        size,
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    ).to(tt_device)

    return tt_tensor


def tt2torch_tensor(tt_tensor):
    host = tt_lib.device.GetHost()
    tt_output = tt_tensor.to(host)
    py_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())
    return py_output


def model_location_generator(rel_path):
    internal_weka_path = Path("/mnt/MLPerf")
    has_internal_weka = (internal_weka_path / "bit_error_tests").exists()

    if has_internal_weka:
        return Path("/mnt/MLPerf") / rel_path
    else:
        return Path("/opt/tt-metal-models") / rel_path


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(
            f"WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}"
        )
    return new_size


IMG_FORMATS = (
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
)  # include image suffixes
VID_FORMATS = (
    "asf",
    "avi",
    "gif",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpg",
    "ts",
    "wmv",
)  # include video suffixes


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


class LoadImages:
    # image/video dataloader
    def __init__(
        self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1
    ):
        if (
            isinstance(path, str) and Path(path).suffix == ".txt"
        ):  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if "*" in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, "*.*"))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f"{p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, (
            f"No images or videos found in {p}. "
            f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: "

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f"Image Not Found {path}"
            s = f"image {self.count}/{self.nf} {path}: "

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[
                0
            ]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(
            self.cap.get(cv2.CAP_PROP_ORIENTATION_META)
        )  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.nf  # number of files
