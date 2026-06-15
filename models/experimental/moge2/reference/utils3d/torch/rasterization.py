from typing import *
import threading

import torch
from torch import Tensor
import torch.nn.functional as F

try:
    import nvdiffrast.torch as dr
except ImportError:
    raise ImportError("nvdiffrast is not installed. Please install nvdiffrast to use the rasterization functions of utils3d.")

from .transforms import extrinsics_to_view, intrinsics_to_perspective

__all__ = [
    'RastContext',
    'rasterize_triangles', 
    'rasterize_triangles_peeling',
    'sample_texture',
    'texture_composite',
]


class RastContext:
    """
    Create a rasterization context. Nothing but a wrapper of nvdiffrast.torch.RasterizeCudaContext or nvdiffrast.torch.RasterizeGLContext.
    """

    default_backend: Literal['cuda', 'gl'] = 'cuda'
    "Default backend for the default context."

    _threading_local: threading.local = threading.local()
    "Thread-local static variable to store default context."

    def __init__(self, nvd_ctx: Union[dr.RasterizeCudaContext, dr.RasterizeGLContext] = None, *, backend: Literal['cuda', 'gl'] = 'cuda',  device: Union[str, torch.device] = None):
        if nvd_ctx is not None:
            self.nvd_ctx = nvd_ctx
            return 
        
        if backend == 'gl':
            self.nvd_ctx = dr.RasterizeGLContext(device=device)
        elif backend == 'cuda':
            self.nvd_ctx = dr.RasterizeCudaContext(device=device)
        else:
            raise ValueError(f'Unknown backend: {backend}')
    
    @staticmethod
    def get_default_context() -> 'RastContext':
        """
        Get the thread-local default rasterization context.
        """
        if not hasattr(RastContext._threading_local, 'default_context'):
            RastContext._threading_local.default_context = RastContext(backend=RastContext.default_backend)
        return RastContext._threading_local.default_context


def rasterize_triangles(
    size: Tuple[int, int],
    *,
    vertices: Tensor,
    attributes: Optional[Tensor] = None,
    faces: Tensor,
    view: Tensor = None,
    projection: Tensor = None,
    extrinsics: Tensor = None,
    intrinsics: Tensor = None,
    near: float = 0.01,
    far: float = float('inf'),
    return_image_derivatives: bool = False,
    return_depth: bool = False,
    return_interpolation: bool = False,
    antialiasing: bool = False,
    ctx: Optional[RastContext] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Rasterize triangles.

    Parameters
    ----
        size (Tuple[int, int]): (height, width) of the output image
        vertices (np.ndarray): (B, N, 2 or 3 or 4)
        faces (Tensor): (T, 3)
        attributes (Tensor, optional): (B, N, C) vertex attributes. Defaults to None.
        texture (Tensor, optional): (B, C, H, W) texture. Defaults to None.
        view | extrinsics (Tensor, optional): ([B,] 4, 4) view matrix or extrinsics matrix. Provide either one of them. Defaults to identity.
        projection | intrinsics (Tensor, optional): ([B,] 4, 4) projection matrix or ([B,] 3, 3) intrinsics matrix. Provide either one of them. Defaults to identity.
        near (float, optional): near plane. Defaults to 0.01. Only used for intrinsics. Ignored if projection matrix is provided.
        far (float, optional): far plane. Defaults to inf. Only used for intrinsics. Ignored if projection matrix is provided.
        return_image_derivatives (bool, optional): whether to return screen space derivatives of the attributes. Defaults to False.
        return_depth (bool, optional): whether to return depth map. Defaults to False.
        return_interpolation (bool, optional): whether to return triangle interpolation maps. Defaults to False.
        antialiasing (Union[bool, List[int]], optional): whether to perform antialiasing. Defaults to True. If a list of indices is provided, only those channels will be antialiased.
        ctx (RastContext): rasterization context. Defaults to the thread-local default context. If custom context is needed, provide one with utils3d.pt.RastContext().

    Returns
    ----
    A dictionary containing:
        - image: (Tensor): (B, H, W, C)
        - image_dr: (Tensor): (B, H, W, C * 2) screen space derivatives of the attributes
        - depth: (Tensor): (B, H, W) Linear depth. Empty pixels have depth inf.
        - mask: (torch.BoolTensor): (B, H, W) mask of valid pixels
        - interpolation_id: (Tensor): (B, H, W) triangle ID map. For empty pixels, the value is -1.
        - interpolation_uv: (Tensor): (B, H, W, 2) triangle UV (first two channels of barycentric coordinates)
    """
    if ctx is None:
        ctx = RastContext.get_default_context()

    assert vertices.ndim == 3
    assert faces.ndim == 2

    if vertices.shape[-1] == 2:
        vertices = torch.cat([vertices, torch.zeros_like(vertices[..., :1]), torch.ones_like(vertices[..., :1])], dim=-1)
    elif vertices.shape[-1] == 3:
        vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
    elif vertices.shape[-1] == 4:
        pass
    else:
        raise ValueError(f'Wrong shape of vertices: {vertices.shape}')
    
    if intrinsics is not None:
        assert projection is None, "Cannot specify both intrinsics and projection."
        projection = intrinsics_to_perspective(intrinsics, near=near, far=far)
    if extrinsics is not None:
        assert view is None, "Cannot specify both extrinsics and view."
        view = extrinsics_to_view(extrinsics)

    mvp = projection if projection is not None else torch.eye(4).to(vertices)
    if view is not None:
        mvp = mvp @ view

    pos_clip = vertices @ mvp.transpose(-1, -2)
    faces = faces.to(torch.int32).contiguous()
    if attributes is not None:
        attributes = attributes.contiguous()

    # Rasterize
    rast_out, rast_db = dr.rasterize(ctx.nvd_ctx, pos_clip, faces, resolution=size, grad_db=True)

    # Process outputs
    mask = (rast_out[..., 3] > 0).flip(1)

    if attributes is not None:
        image, image_dr = dr.interpolate(attributes, rast_out, faces, rast_db, diff_attrs='all' if return_image_derivatives else None)
        if antialiasing:
            image = dr.antialias(image, rast_out, pos_clip, faces)
        image = image.flip(1)
        
        if return_image_derivatives is not None:
            image_dr = image_dr.flip(1)
        else:
            image_dr = None
    else:
        image = None
        image_dr = None

    if return_interpolation:
        interpolation_id = rast_out[..., 3].flip(1).long() - 1
        interpolation_uv = rast_out[..., :2].flip(1)
    else:
        interpolation_id = None
        interpolation_uv = None

    if return_depth:
        ndc_coord_map = torch.cat([
            F.affine_grid(
                torch.tensor([[[1., 0., 0.], [0., -1., 0.]]], device=rast_out.device, dtype=rast_out.dtype), 
                (rast_out.shape[0], 1, size[0], size[1]), 
                align_corners=False
            ), 
            rast_out[..., 2:3].flip(1),
            torch.ones((*rast_out.shape[:3], 1), device=rast_out.device, dtype=rast_out.dtype)
        ], dim=-1)
        view_coord_map = ndc_coord_map @ torch.linalg.inv(projection[..., None, :, :]).mT
        depth = -view_coord_map[..., 2] / view_coord_map[..., 3]
        depth = torch.where(mask, depth, torch.inf)
    else:
        depth = None

    output = {
        'image': image,
        'image_dr': image_dr,
        'depth': depth,
        'mask': mask,
        'interpolation_id': interpolation_id,
        'interpolation_uv': interpolation_uv,
    }

    output = {k: v for k, v in output.items() if v is not None}

    return output


def rasterize_triangles_peeling(
    size: Tuple[int, int],
    *,
    vertices: Tensor,
    attributes: Optional[Tensor] = None,
    faces: Tensor,
    view: Tensor = None,
    projection: Tensor = None,
    extrinsics: Tensor = None,
    intrinsics: Tensor = None,
    near: float = 0.01,
    far: float = float('inf'),
    return_image_derivatives: bool = False,
    return_depth: bool = False,
    return_interpolation: bool = False,
    antialiasing: bool = False,
    ctx: Optional[RastContext] = None,
) -> Iterator[Iterator[Dict[str, Tensor]]]:
    """
    Rasterize a mesh with vertex attributes using depth peeling.

    Parameters
    ----
        size (Tuple[int, int]): (height, width) of the output image
        vertices (np.ndarray): (B, N, 2 or 3 or 4)
        faces (Tensor): (T, 3)
        attributes (Tensor, optional): (B, N, C) vertex attributes. Defaults to None.
        texture (Tensor, optional): (B, C, H, W) texture. Defaults to None.
        view | extrinsics (Tensor, optional): ([B,] 4, 4) view matrix or extrinsics matrix. Provide either one of them. Defaults to identity.
        projection | intrinsics (Tensor, optional): ([B,] 4, 4) projection matrix or ([B,] 3, 3) intrinsics matrix. Provide either one of them. Defaults to identity.
        near (float, optional): near plane. Defaults to 0.01. Only used for intrinsics. Ignored if projection matrix is provided.
        far (float, optional): far plane. Defaults to inf. Only used for intrinsics. Ignored if projection matrix is provided.
        return_image_derivatives (bool, optional): whether to return screen space derivatives of the attributes. Defaults to False.
        return_depth (bool, optional): whether to return depth map. Defaults to False.
        return_interpolation (bool, optional): whether to return triangle interpolation maps. Defaults to False.
        antialiasing (Union[bool, List[int]], optional): whether to perform antialiasing. Defaults to True. If a list of indices is provided, only those channels will be antialiased.
        ctx (RastContext): rasterization context. Defaults to the thread-local default context. If custom context is needed, provide one with utils3d.pt.RastContext().

    Returns
    ----
    A generator of dictionaries for each layer containing:
        - mask: (List[torch.BoolTensor]): (B, H, W) mask of valid pixels in this layer
        - image: (List[Tensor]): (B, C, H, W) rendered images
        - image_dr: (List[Tensor]): (B, *, H, W) screen space derivatives of the attributes
        - depth: (List[Tensor]): (B, H, W) linear depth. Empty pixels have depth inf.
        - interpolation_id: (List[Tensor]): (B, H, W) triangle ID map. For empty pixels, the value is -1.
        - interpolation_uv: (List[Tensor]): (B, H, W, 2) triangle UV (first two channels of barycentric coordinates)
    
    The last layer yielded will be empty, then the generator will stop.

    Example
    ----
    ```
    for i, layer_output in enumerate(rasterize_triangles_peeling(
        (512, 512), 
        vertices=vertices, 
        faces=faces, 
        attributes=attributes,
        view=view,
        projection=projection
    )):
        print(f"Layer {i}:")
        for key, value in layer_output.items():
            print(f"  {key}: {value.shape}")
        if i >= 4:  # Stop after 5 layers at most
            break
    ```
    """
    if ctx is None:
        ctx = RastContext.get_default_context()

    assert vertices.ndim == 3
    assert faces.ndim == 2

    if vertices.shape[-1] == 2:
        vertices = torch.cat([vertices, torch.zeros_like(vertices[..., :1]), torch.ones_like(vertices[..., :1])], dim=-1)
    elif vertices.shape[-1] == 3:
        vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
    elif vertices.shape[-1] == 4:
        pass
    else:
        raise ValueError(f'Wrong shape of vertices: {vertices.shape}')
    
    if intrinsics is not None:
        assert projection is None, "Cannot specify both intrinsics and projection."
        projection = intrinsics_to_perspective(intrinsics, near=near, far=far)
    if extrinsics is not None:
        assert view is None, "Cannot specify both extrinsics and view."
        view = extrinsics_to_view(extrinsics)

    mvp = projection if projection is not None else torch.eye(4).to(vertices)
    if view is not None:
        mvp = mvp @ view

    pos_clip = vertices @ mvp.mT
    faces = faces.contiguous()
    if attributes is not None:
        attributes = attributes.contiguous()
        
    with dr.DepthPeeler(ctx.nvd_ctx, pos_clip, faces, resolution=size) as peeler:
        while True:
            # Rasterize
            rast_out, rast_db = peeler.rasterize_next_layer()
            
            # Process outputs
            mask = (rast_out[..., 3] > 0).flip(1)

            if attributes is not None:
                image, image_dr = dr.interpolate(attributes, rast_out, faces, rast_db, diff_attrs='all' if return_image_derivatives else None)
                if antialiasing:
                    image = dr.antialias(image, rast_out, pos_clip, faces)
                image = image.flip(1).permute(0, 3, 1, 2)
                    
                if return_image_derivatives is not None:
                    image_dr = image_dr.flip(1).permute(0, 3, 1, 2)
                else:
                    image_dr = None
            else:
                image = None
                image_dr = None

            if return_interpolation:
                interpolation_id = rast_out[..., 3].flip(1).long() - 1
                interpolation_uv = rast_out[..., :2].flip(1)
            else:
                interpolation_id = None
                interpolation_uv = None

            if return_depth:
                ndc_coord_map = torch.cat([
                    F.affine_grid(
                        torch.tensor([[[1., 0., 0.], [0., -1., 0.]]], device=rast_out.device, dtype=rast_out.dtype), 
                        (rast_out.shape[0], 1, size[0], size[1]), 
                        align_corners=False
                    ), 
                    rast_out[..., 2:3].flip(1),
                    torch.ones((*rast_out.shape[:3], 1), device=rast_out.device, dtype=rast_out.dtype)
                ], dim=-1)
                view_coord_map = ndc_coord_map @ torch.linalg.inv(projection[..., None, :, :]).mT
                depth = -view_coord_map[..., 2] / view_coord_map[..., 3]
                depth = torch.where(mask, depth, torch.inf)
            else:
                depth = None

            output = {
                'image': image,
                'image_dr': image_dr,
                'depth': depth,
                'mask': mask,
                'interpolation_id': interpolation_id,
                'interpolation_uv': interpolation_uv,
            }
            output = {k: v for k, v in output.items() if v is not None}
            yield output

            if not mask.any():
                break

def sample_texture(
    texture: Tensor,
    uv: Tensor,
    uv_dr: Tensor,
) -> Tensor:
    """
    Interpolate texture using uv coordinates.
    
    ## Parameters
        texture (Tensor): (B, H, W, C) texture
        uv (Tensor): (B, H, W, 2) uv coordinates
        uv_dr (Tensor): (B, H, W, 4) uv derivatives
        
    ## Returns
        Tensor: (B, H, W, C) interpolated texture
    """
    texture = texture.flip(2).contiguous()
    return dr.texture(texture, uv, uv_dr).flip(1)
    
    
def texture_composite(
    texture: Tensor,
    uv: List[Tensor],
    uv_da: List[Tensor],
    background: Tensor = None,
) -> Tuple[Tensor, Tensor]:
    """
    Composite textures with depth peeling output.
    
    ## Parameters
        texture (Tensor): (B, C+1, H, W) texture
            NOTE: the last channel is alpha channel
        uv (List[Tensor]): list of (B, H, W, 2) uv coordinates
        uv_da (List[Tensor]): list of (B, H, W, 4) uv derivatives
        background (Optional[Tensor], optional): (B, C, H, W) background image. Defaults to None (black).
        
    ## Returns
        image: (Tensor): (B, C, H, W) rendered image
        alpha: (Tensor): (B, H, W) alpha channel
    """
    assert len(uv) == len(uv_da)
    if background is not None:
        assert texture.shape[1] == background.shape[1] + 1
    
    C = texture.shape[1] - 1
    B, H, W = uv[0].shape[:3]
    texture = texture.flip(2).permute(0, 2, 3, 1).contiguous()
    alpha = torch.zeros(B, H, W, device=texture.device)
    if background is None:
        image = torch.zeros(B, H, W, C, device=texture.device)
    else:
        image = background.clone().permute(0, 2, 3, 1)      # [B, H, W, C]
    for i in range(len(uv)):
        texture_map = dr.texture(texture, uv[i], uv_da[i])      # [B, H, W, C+1]
        _alpha = texture_map[..., -1]                           # [B, H, W]
        _weight = _alpha * (1 - alpha)                          # [B, H, W]
        image = image + texture_map[..., :-1] * _weight.unsqueeze(-1)   # [B, H, W, C]
        alpha = alpha + _weight                         # [B, H, W]
    return image.flip(1).permute(0, 3, 1, 2), alpha.flip(1)


# def warp_image_by_depth(
#     depth: torch.FloatTensor,
#     image: torch.FloatTensor = None,
#     mask: torch.BoolTensor = None,
#     width: int = None,
#     height: int = None,
#     *,
#     extrinsics_src: torch.FloatTensor = None,
#     extrinsics_tgt: torch.FloatTensor = None,
#     intrinsics_src: torch.FloatTensor = None,
#     intrinsics_tgt: torch.FloatTensor = None,
#     near: float = 0.1,
#     far: float = 100.0,
#     antialiasing: bool = True,
#     backslash: bool = False,
#     padding: int = 0,
#     return_uv: bool = False,
#     return_dr: bool = False,
#     ctx: Optional[RastContext] = None,
# ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.BoolTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
#     """
#     Warp image by depth. 
#     NOTE: if batch size is 1, image mesh will be triangulated aware of the depth, yielding less distorted results.
#     Otherwise, image mesh will be triangulated simply for batch rendering.

#     ## Parameters
#         depth (Tensor): (B, H, W) linear depth
#         image (Tensor): (B, C, H, W). None to use image space uv. Defaults to None.
#         width (int, optional): width of the output image. None to use the same as depth. Defaults to None.
#         height (int, optional): height of the output image. Defaults the same as depth..
#         extrinsics_src (Tensor, optional): (B, 4, 4) extrinsics matrix for source. None to use identity. Defaults to None.
#         extrinsics_tgt (Tensor, optional): (B, 4, 4) extrinsics matrix for target. None to use identity. Defaults to None.
#         intrinsics_src (Tensor, optional): (B, 3, 3) intrinsics matrix for source. None to use the same as target. Defaults to None.
#         intrinsics_tgt (Tensor, optional): (B, 3, 3) intrinsics matrix for target. None to use the same as source. Defaults to None.
#         near (float, optional): near plane. Defaults to 0.1. 
#         far (float, optional): far plane. Defaults to 100.0.
#         antialiasing (bool, optional): whether to perform antialiasing. Defaults to True.
#         backslash (bool, optional): whether to use backslash triangulation. Defaults to False.
#         padding (int, optional): padding of the image. Defaults to 0.
#         return_uv (bool, optional): whether to return the uv. Defaults to False.
#         return_dr (bool, optional): whether to return the image-space derivatives of uv. Defaults to False.
#         ctx (RastContext): rasterization context
    
#     ## Returns
#         image: (torch.FloatTensor): (B, C, H, W) rendered image
#         depth: (torch.FloatTensor): (B, H, W) linear depth, ranging from 0 to inf
#         mask: (torch.BoolTensor): (B, H, W) mask of valid pixels
#         uv: (torch.FloatTensor): (B, 2, H, W) image-space uv
#         dr: (torch.FloatTensor): (B, 4, H, W) image-space derivatives of uv
#     """
#     if ctx is None:
#         ctx = RastContext.get_default_context()

#     assert depth.ndim == 3
#     batch_size = depth.shape[0]

#     if width is None:
#         width = depth.shape[-1]
#     if height is None:
#         height = depth.shape[-2]
#     if image is not None:
#         assert image.shape[-2:] == depth.shape[-2:], f'Shape of image {image.shape} does not match shape of depth {depth.shape}'

#     if extrinsics_src is None:
#         extrinsics_src = torch.eye(4).to(depth)
#     if extrinsics_tgt is None:
#         extrinsics_tgt = torch.eye(4).to(depth)
#     if intrinsics_src is None:
#         intrinsics_src = intrinsics_tgt
#     if intrinsics_tgt is None:
#         intrinsics_tgt = intrinsics_src
    
#     assert all(x is not None for x in [extrinsics_src, extrinsics_tgt, intrinsics_src, intrinsics_tgt]), "Make sure you have provided all the necessary camera parameters."

#     view_tgt = transforms.extrinsics_to_view(extrinsics_tgt)
#     perspective_tgt = transforms.intrinsics_to_perspective(intrinsics_tgt, near=near, far=far)

#     if padding > 0:
#         uv, faces = utils.image_mesh(width=width+2, height=height+2)
#         uv = (uv - 1 / (width + 2)) * ((width + 2) / width)
#         uv_ = uv.clone().reshape(height+2, width+2, 2)
#         uv_[0, :, 1] -= padding / height
#         uv_[-1, :, 1] += padding / height
#         uv_[:, 0, 0] -= padding / width
#         uv_[:, -1, 0] += padding / width
#         uv_ = uv_.reshape(-1, 2)
#         depth = torch.nn.functional.pad(depth, [1, 1, 1, 1], mode='replicate')
#         if image is not None:
#             image = torch.nn.functional.pad(image, [1, 1, 1, 1], mode='replicate')
#         uv, uv_, faces = uv.to(depth.device), uv_.to(depth.device), faces.to(depth.device)
#         pts = transforms.unproject_cv(
#             uv_,
#             depth.flatten(-2, -1),
#             extrinsics_src,
#             intrinsics_src,
#         )
#     else:    
#         uv, faces = utils.image_mesh(width=depth.shape[-1], height=depth.shape[-2])
#         if mask is not None:
#             depth = torch.where(mask, depth, Tensor(far, dtype=depth.dtype, device=depth.device))
#         uv, faces = uv.to(depth.device), faces.to(depth.device)
#         pts = transforms.unproject_cv(
#             uv,
#             depth.flatten(-2, -1),
#             extrinsics_src,
#             intrinsics_src,
#         )

#     # triangulate
#     if batch_size == 1:
#         faces = mesh.triangulate_mesh(faces, vertices=pts[0])
#     else:
#         faces = mesh.triangulate_mesh(faces, backslash=backslash)

#     # rasterize attributes
#     diff_attrs = None
#     if image is not None:
#         attr = image.permute(0, 2, 3, 1).flatten(1, 2)
#         if return_dr or return_uv:
#             if return_dr:
#                 diff_attrs = [image.shape[1], image.shape[1]+1]
#             if return_uv and antialiasing:
#                 antialiasing = list(range(image.shape[1]))
#             attr = torch.cat([attr, uv.expand(batch_size, -1, -1)], dim=-1)
#     else:
#         attr = uv.expand(batch_size, -1, -1)
#         if antialiasing:
#             print("\033[93mWarning: you are performing antialiasing on uv. This may cause artifacts.\033[0m")
#         if return_uv:
#             return_uv = False
#             print("\033[93mWarning: image is None, return_uv is ignored.\033[0m")
#         if return_dr:
#             diff_attrs = [0, 1]

#     if mask is not None:
#         attr = torch.cat([attr, mask.float().flatten(1, 2).unsqueeze(-1)], dim=-1)

#     rast = rasterize_triangle_faces(
#         ctx,
#         pts,
#         faces,
#         width,
#         height,
#         attr=attr,
#         view=view_tgt,
#         perspective=perspective_tgt,
#         antialiasing=antialiasing,
#         diff_attrs=diff_attrs,
#     )
#     if return_dr:
#         output_image, screen_depth, output_dr = rast['image'], rast['depth'], rast['image_dr']
#     else:
#         output_image, screen_depth = rast['image'], rast['depth']
#     output_mask = screen_depth < 1.0

#     if mask is not None:
#         output_image, rast_mask = output_image[..., :-1, :, :], output_image[..., -1, :, :]
#         output_mask &= (rast_mask > 0.9999).reshape(-1, height, width)

#     if (return_dr or return_uv) and image is not None:
#         output_image, output_uv = output_image[..., :-2, :, :], output_image[..., -2:, :, :]

#     output_depth = transforms.depth_buffer_to_linear(screen_depth, near=near, far=far) * output_mask
#     output_image = output_image * output_mask.unsqueeze(1)

#     outs = [output_image, output_depth, output_mask]
#     if return_uv:
#         outs.append(output_uv)
#     if return_dr:
#         outs.append(output_dr)
#     return tuple(outs)


# def warp_image_by_forward_flow(
#     image: torch.FloatTensor,
#     flow: torch.FloatTensor,
#     depth: torch.FloatTensor = None,
#     *,
#     antialiasing: bool = True,
#     backslash: bool = False,
#     ctx: Optional[RastContext] = None,
# ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
#     """
#     Warp image by forward flow.
#     NOTE: if batch size is 1, image mesh will be triangulated aware of the depth, yielding less distorted results.
#     Otherwise, image mesh will be triangulated simply for batch rendering.

#     ## Parameters
#         ctx (Union[dr.RasterizeCudaContext, dr.RasterizeGLContext]): rasterization context
#         image (Tensor): (B, C, H, W) image
#         flow (Tensor): (B, 2, H, W) forward flow
#         depth (Tensor, optional): (B, H, W) linear depth. If None, will use the same for all pixels. Defaults to None.
#         antialiasing (bool, optional): whether to perform antialiasing. Defaults to True.
#         backslash (bool, optional): whether to use backslash triangulation. Defaults to False.
    
#     ## Returns
#         image: (torch.FloatTensor): (B, C, H, W) rendered image
#         mask: (torch.BoolTensor): (B, H, W) mask of valid pixels
#     """
#     if ctx is None:
#         ctx = RastContext.get_default_context()
        
#     assert image.ndim == 4, f'Wrong shape of image: {image.shape}'
#     batch_size, _, height, width = image.shape

#     if depth is None:
#         depth = torch.ones_like(flow[:, 0])

#     extrinsics = torch.eye(4).to(image)
#     fov = torch.deg2rad(Tensor([45.0], device=image.device))
#     intrinsics = transforms.intrinsics_from_fov(fov, width, height, normalize=True)[0] 
   
#     view = transforms.extrinsics_to_view(extrinsics)
#     perspective = transforms.intrinsics_to_perspective(intrinsics, near=0.1, far=100)

#     uv, faces = build_mesh_from_map(uv_map(width=width, height=height))
#     uv, faces = uv.to(image.device), faces.to(image.device)
#     uv = uv + flow.permute(0, 2, 3, 1).flatten(1, 2)
#     pts = transforms.unproject_cv(
#         uv,
#         depth.flatten(-2, -1),
#         extrinsics,
#         intrinsics,
#     )

#     # triangulate
#     if batch_size == 1:
#         faces = mesh.triangulate_mesh(faces, vertices=pts[0])
#     else:
#         faces = mesh.triangulate_mesh(faces, backslash=backslash)

#     # rasterize attributes
#     attr = image.permute(0, 2, 3, 1).flatten(1, 2)
#     rast = rasterize_triangles(
#         ctx,
#         width,
#         height,
#         vertices=pts,
#         attr=attr,
#         faces=faces,
#         view=view,
#         perspective=perspective,
#         antialiasing=antialiasing,
#     )
#     output_image, screen_depth = rast['image'], rast['depth']
#     output_mask = screen_depth < 1.0
#     output_image = output_image * output_mask.unsqueeze(1)

#     outs = [output_image, output_mask]
#     return tuple(outs)
